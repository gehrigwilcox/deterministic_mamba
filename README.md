# deterministic_mamba

Patches that make **Mamba1, Mamba2 and Mamba3** training bitwise reproducible, on top of upstream
[`mamba_ssm`](https://github.com/state-spaces/mamba) and
[`causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d). The patched packages keep their
upstream names, so your imports don't change.

**Default behavior is upstream's**: the fast kernels, with atomics and autotuning. The
deterministic kernels are used only when you ask PyTorch for determinism:

```python
import torch
from mamba_ssm.models.mixer_seq_simple import MixerModel

torch.use_deterministic_algorithms(True)   # any time, including after the import
# export CUBLAS_WORKSPACE_CONFIG=:4096:8   # needed by PyTorch's own GEMMs in this mode

model = MixerModel(..., ssm_cfg={"layer": "Mamba1"})   # or "Mamba2", or "Mamba3"
```

Other ways to switch: `mamba_ssm.utils.determinism.set_deterministic_mode(True|False|None)`
(process override), or `MAMBA_DETERMINISTIC=1` (and `CAUSAL_CONV1D_DETERMINISTIC=1` for the
conv). `torch.use_deterministic_algorithms` controls both packages at once.

Pinned versions (`versions.env`): **mamba_ssm v2.3.2.post1** and **causal-conv1d v1.7.0**.

## Install

```bash
# Use the Python env you train in; nvcc must match torch's CUDA major version.
PIP="python -m pip" CUDA_ARCHS="80;90" scripts/build.sh
```

`build.sh` clones the pinned tags and applies `patches/` with `git am`. It then pip-installs
`causal_conv1d` and `mamba_ssm` from source. `CUDA_ARCHS` is optional and only cuts compile
time; without it you get upstream's full architecture list. `--patch-only` stops after patching
(the sources are left in `build/`).

## What upstream already covers and what the patches add

Upstream added a deterministic mode in mamba_ssm 2.3 and causal-conv1d 1.7. These patches fill
the gaps that remain:

| Component | Upstream 2.3.2.post1 / 1.7.0 | This repo |
|---|---|---|
| Mamba2 SSD Triton kernels (atomics) | deterministic mode ✔ | unchanged |
| causal-conv1d backward (atomics, both layouts) | deterministic mode ✔ | **0001**: fixes a SiLU gradient bug (below) |
| **Mamba1** CUDA selective-scan backward | atomics only, no deterministic mode ✘ | **mamba 0001**: per-block workspace plus fixed-order reduction |
| Autotuned Triton kernels (config choice changes reduction order) | config set is chosen when the module is *imported*; Mamba3 kernels never restricted ✘ | **mamba 0002**: config set chosen at each *launch*; covers every autotuned kernel, Mamba3 included |
| **Mamba3** SISO backward, dK state post-step (atomics across GQA heads and batch) | atomics only ✘ | **mamba 0003**: workspace plus fixed-order reduction |
| `MixerModel` with Mamba3 layers | not accepted ✘ | **mamba 0004**: `ssm_cfg={"layer": "Mamba3"}` |

Notes:

* **mamba 0002 matters even for Mamba2.** Upstream's `autotune_configs()` runs when the module
  is imported. If you call `torch.use_deterministic_algorithms(True)` after `import mamba_ssm`
  (the usual order), full autotuning stays on, and different processes can choose different
  kernel configs. After the patch the choice follows the switch at every launch, and both modes
  work in one process.
* **causal-conv1d 0001 is a correctness fix.** It applies in both modes. In the channel-first
  SiLU backward that Mamba1 uses, thread 0 bounds-checked the wrong index when loading the tail
  of the previous chunk. When the last chunk has fewer than `kNElts` elements (for example
  seqlen 2049 or 4097), `dx`, `dweight` and `dbias` came out wrong at that boundary.
* **Mamba3 MIMO** runs on TileLang kernels, which use fixed configs and no atomics. Its Triton
  helper kernels are covered by 0002. The other Mamba3 SISO atomics (`dDT` and `dTrap`
  first-token terms, `dAngles`) touch each address exactly once, so they are already
  deterministic.

## Cost of deterministic mode

Mamba1 selective-scan backward (GTX 1660 SUPER, batch 8, d_inner 1536, L 2048, d_state 16, bf16):
about 1.4× the kernel time (28.8 → 40.7 ms), with about 260 MiB of extra workspace at the
default 256 MiB budget (`MAMBA_DETERMINISTIC_WORKSPACE_MB`). The budget never changes the bits.
For the Triton kernels, deterministic mode uses one fixed config per kernel instead of the
autotuned best. `TRITON_CACHE_AUTOTUNING=1` (Triton ≥ 3.4) instead autotunes once and caches the
choice on disk, which is reproducible as long as every run shares that cache.

Reproducibility holds for a given GPU model and software stack (driver, CUDA, torch, triton).

## Tests (`tests/`, run against the installed patched packages)

```bash
pytest tests
```

* `test_mamba1.py`: correctness against the reference in both modes (fp32/fp16/bf16, complex,
  grouped/variable/constant B and C), bitwise repeatability, all three ways to switch the mode,
  budget independence, the conv boundary fix, and that default mode is still upstream's atomic
  path.
* `test_mamba2_mamba3.py`: Mamba2 SSD and Mamba3 SISO are bitwise repeatable and close to
  default mode (GQA and final-state gradients included). Also checks that switching after
  import selects the deterministic configs.
* `test_mixer_model.py`: `MixerModel` training with Mamba1/2/3 layers gives identical weights
  in-process and across two separate processes, with determinism switched on *after* import.

Upstream's own `tests/test_determinism.py` (Mamba2) also passes with these patches.

## Moving to a new upstream version

```bash
git clone --branch <new-tag> https://github.com/state-spaces/mamba work/mamba
cd work/mamba && git checkout -b deterministic && git am ../../patches/mamba/*.patch   # resolve conflicts
# build (scripts/build.sh --from-work), run pytest tests, then:
scripts/export_patches.sh    # rewrites patches/ from the commits on top of the tag
# and bump the tags in versions.env
```

If a patch no longer applies, `scripts/dev/edit_*.py` are the scripted edits that produced each
patch. Every anchor in them is asserted, so they show exactly which upstream code moved. Check
first whether upstream has started covering a component itself, and drop that patch if so.

## Layout

```
versions.env              pinned upstream tags
patches/mamba/            0001-0004 (git format-patch)
patches/causal-conv1d/    0001
scripts/build.sh          fetch + patch + install
scripts/export_patches.sh regenerate patches/ from work/
scripts/dev/              edit scripts that produced the patches; limit_archs.py (build helper)
tests/                    determinism + correctness tests
```

The git history also holds the earlier standalone fork for mamba_ssm 2.2.4 (Mamba1 only).
