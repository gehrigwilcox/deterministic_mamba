"""One-off edit script used to author patches/mamba/0002 (call-time autotune selection).
Run from an upstream mamba checkout."""
import glob
import re

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

# ------------------------------------------------------------- determinism.py
p = "mamba_ssm/utils/determinism.py"
s = open(p).read()
s = sub(s, '''def autotune_configs(configs):
    """Select autotune configs for deterministic mode.
    
    Uses cached autotuning (TRITON_CACHE_AUTOTUNING=1) if Triton >= 3.4.0,
    otherwise auto-selects the cheapest config by block size * stages.
    """
    if not configs or not use_deterministic_mode():
        return configs
''', '''def deterministic_configs(configs):
    """The configs to autotune over in deterministic mode (regardless of the current mode).

    Uses cached autotuning (TRITON_CACHE_AUTOTUNING=1) if Triton >= 3.4.0,
    otherwise auto-selects the cheapest config by block size * stages.
    """
    if not configs:
        return configs
''')
s = sub(s, '''    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]
''', '''    filtered = _filter_configs_by_block_sizes(configs)
    if filtered:
        return filtered
    return [min(configs, key=_estimate_config_cost)]


def autotune_configs(configs):
    """Kept for backward compatibility. Evaluated once, at decoration (import) time, so it only
    takes effect if deterministic mode is already on when the module is imported. Kernels in
    mamba_ssm use `autotune` below instead, which decides at every launch."""
    if not configs or not use_deterministic_mode():
        return configs
    return deterministic_configs(configs)


class _ModeAwareAutotuner:
    """A drop-in for a `triton.autotune`-decorated kernel that picks its config set at *launch*
    time: full autotuning normally, `deterministic_configs(...)` when use_deterministic_mode() is
    true. So `torch.use_deterministic_algorithms(True)` takes effect even when called after
    `import mamba_ssm`, and both modes can be used in one process (each keeps its own cache)."""

    def __init__(self, fn, configs, kwargs):
        self._fn = fn
        self._all_configs = list(configs)
        self._kwargs = kwargs
        self._tuned = triton.autotune(configs=self._all_configs, **kwargs)(fn)
        self._det = None
        self._last = self._tuned

    def _select(self):
        if use_deterministic_mode():
            if self._det is None:
                self._det = triton.autotune(configs=deterministic_configs(self._all_configs), **self._kwargs)(self._fn)
            self._last = self._det
        else:
            self._last = self._tuned
        return self._last

    def __getitem__(self, grid):
        return self._select()[grid]

    def run(self, *args, **kwargs):
        return self._select().run(*args, **kwargs)

    @property
    def configs(self):
        # The full list, in both modes: callers size workspaces from e.g. min(BLOCK_SIZE_N), and the
        # deterministic configs are a subset, so sizes derived from the full list stay valid.
        return self._tuned.configs

    @property
    def best_config(self):
        # The config used by the most recent launch (in whichever mode it ran).
        return self._last.best_config

    def __getattr__(self, name):
        return getattr(self._tuned, name)


def autotune(configs, **kwargs):
    """Like `triton.autotune`, but deterministic mode is checked at each launch (see
    _ModeAwareAutotuner). Accepts the same arguments."""
    def decorator(fn):
        return _ModeAwareAutotuner(fn, configs, kwargs)
    return decorator
''')
open(p, "w").write(s)

# ------------------------------------------------------------- kernels
def strip_autotune_configs(src):
    """configs=autotune_configs(X) -> configs=X (balanced parentheses)."""
    out, i = [], 0
    key = "autotune_configs("
    while True:
        j = src.find(key, i)
        if j < 0:
            out.append(src[i:]); break
        # skip the import line / definitions
        line_start = src.rfind("\n", 0, j) + 1
        if src[line_start:j].strip().startswith(("from", "import", "def")) or src[j - 1] in "_,( " and "import" in src[line_start:j]:
            out.append(src[i:j + len(key)]); i = j + len(key); continue
        depth, k = 1, j + len(key)
        while depth:
            depth += {"(": 1, ")": -1}.get(src[k], 0); k += 1
        out.append(src[i:j]); out.append(src[j + len(key):k - 1]); i = k
    return "".join(out)

files = sorted(set(glob.glob("mamba_ssm/ops/triton/*.py") + glob.glob("mamba_ssm/ops/triton/mamba3/*.py")))
changed = []
for p in files:
    s = open(p).read()
    n = s.count("@triton.autotune(")
    if n == 0:
        continue
    s = s.replace("@triton.autotune(", "@autotune(")
    s = strip_autotune_configs(s)
    assert "autotune_configs(" not in s.replace("import", ""), p
    # imports: drop autotune_configs from existing determinism import, add autotune
    s = re.sub(r"(\n\s*)autotune_configs,\n", r"\1autotune,\n", s)
    s = s.replace("from mamba_ssm.utils.determinism import autotune_configs\n", "from mamba_ssm.utils.determinism import autotune\n")
    if not re.search(r"^from mamba_ssm\.utils\.determinism import[^\n]*\bautotune\b|^\s+autotune,$", s, re.M):
        m = re.search(r"^import triton\.language as tl\n", s, re.M)
        assert m, p
        s = s[:m.end()] + "\nfrom mamba_ssm.utils.determinism import autotune\n" + s[m.end():]
    open(p, "w").write(s)
    changed.append((p, n))
for p, n in changed:
    print(f"{n:3d}  {p}")
