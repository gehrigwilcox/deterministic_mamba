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

model = MixerModel(..., ssm_cfg={"layer": "Mamba1"})   # or "Mamba2", or "Mamba3"
```

> **You must also set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the process starts.**
> Deterministic mode makes PyTorch's own matrix multiplies (all the linear layers) refuse to run
> without it. cuBLAS reads the variable once, when CUDA initializes, so setting it later from
> Python (e.g. `os.environ[...]` after the first CUDA call) does nothing. If you miss it, the
> failure is a `RuntimeError` from deep inside a GEMM that names the variable but doesn't say it
> had to be set at startup. Put it in the launch command or job script:
>
> ```bash
> CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py
> ```

Other ways to switch: `mamba_ssm.utils.determinism.set_deterministic_mode(True|False|None)`
(process override), or `MAMBA_DETERMINISTIC=1` (and `CAUSAL_CONV1D_DETERMINISTIC=1` for the
conv). `torch.use_deterministic_algorithms` controls both packages at once.

Pinned versions (`versions.env`): **mamba_ssm v2.3.2.post1** and **causal-conv1d v1.7.0**.

## Prerequisites

These come from upstream mamba_ssm 2.3.2.post1 and apply to the stock package as well:

* **torch ≥ 2.8.0 to `import mamba_ssm` at all, whichever Mamba layer you use.** `mamba_ssm`
  imports its Mamba3 module at package import. That pulls in its hard dependency
  `quack-kernels`, which uses `torch.float4_e2m1fn_x2`, a dtype added in torch 2.8. On older
  torch the import fails with an `AttributeError`, even for plain Mamba1.
* **Triton ≥ 3.5.0 (in practice torch ≥ 2.9) for Mamba3.** Triton ≤ 3.4 requires every `tl.dot`
  dimension to be ≥ 16. Mamba3's SISO forward kernel computes a row sum as a dot with an N = 1
  column of ones, so on Triton 3.4 (which torch 2.8 pins) Mamba3 fails to compile with
  `Input shapes should have M >= 16, N >= 16 and K >= 16`. This happens on every GPU
  architecture. Mamba1 and Mamba2 work on Triton 3.4. Upstream declares `triton>=3.5.0`, but
  `build.sh` installs with `--no-deps` (below), so it checks this itself and warns.
* **A CUDA toolkit (`nvcc`) whose major *and minor* version match `torch.version.cuda`.**
  Matching only the major version isn't enough. For example, nvcc 12.9 against a torch built for
  CUDA 12.1 fails with libcu++ host/device qualifier errors in `<cuda/std/__cmath/traits.h>`.
  `build.sh` refuses a mismatch unless you set `ALLOW_CUDA_MISMATCH=1`.

## Install

```bash
# In the Python env you train in, with CUDA_HOME pointing at the matching toolkit:
PIP="python -m pip" CUDA_ARCHS="80;90" scripts/build.sh
scripts/build.sh --check     # just the environment checks
```

`build.sh` checks the prerequisites above, clones the pinned tags and applies `patches/` with
`git am`. It then builds and installs `causal_conv1d` and `mamba_ssm` from source and runs a
smoke import. `CUDA_ARCHS` is optional and only cuts compile time; without it you get upstream's
full architecture list. `--patch-only` stops after patching (the sources are left in `build/`).

The packages are installed with `pip install --force-reinstall --no-deps --no-cache-dir`:

* `--force-reinstall`: the version string doesn't change between rebuilds, so plain
  `pip install` can skip the build. It then leaves a `.so` compiled against a previous torch,
  which fails at import after a torch upgrade (`undefined symbol: ...c10_cuda_check_implementation...`).
* `--no-deps`: without it, `--force-reinstall` also reinstalls torch and the other dependencies
  from PyPI, replacing your pinned versions.
* `--no-cache-dir`: never reuses a wheel cached from a build against another torch.

Because dependencies aren't installed, `build.sh` lists any missing runtime dependencies
(`einops`, `transformers`, `tilelang==0.1.8`, `apache-tvm-ffi<=0.1.9`, `quack-kernels>=0.3.4`,
...) for you to install. Rerun `build.sh` after every torch upgrade.

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

## Equivalence with upstream

Two questions: does the patched **default** mode behave exactly like stock upstream, and does
**deterministic** mode change results beyond floating-point rounding? Both were measured against
an unpatched build of mamba_ssm 2.3.2.post1 + causal-conv1d 1.7.0. Setup: GTX 1660 SUPER, torch
2.9.1, Triton 3.5.1, fp32. Each run is a separate process, and determinism is switched on after
import. Harness and raw reports: [`tests/equivalence/`](tests/equivalence/).

**1. Patched default mode computes exactly what stock computes.** Two things vary run to run even
in stock: order-dependent atomics, and autotuning, which picks configs by timing in each process.
That makes "stock vs patched" noisy unless both are controlled. So every autotuned Triton kernel
(43) was pinned to the same config in both builds, and the module forward/backward plus a 60-step
`MixerModel` training run were compared tensor by tensor
([`pinned_report.txt`](tests/equivalence/results/pinned_report.txt)):

| | tensors | bit-identical, stock vs stock | bit-identical, stock vs patched default | stock-stable tensors where patched differs |
|---|---|---|---|---|
| Mamba3 | 49 | 49 | **49** (identical 60-step loss curve and weights) | none |
| Mamba2 | 49 | 12 | 14 | none |
| Mamba1 | 54 | 3 | 3 | none |

Mamba3's default training path has no order-dependent atomics, so it is fully bit-identical to
stock. For Mamba1 and Mamba2, every tensor that stock reproduces exactly, the patched build
reproduces exactly too. The rest are the atomic gradients and the training state they feed, and
those vary in stock itself. The patched differences there (max rel 6.9e-4 Mamba1, 2.8e-6 Mamba2)
are no larger than stock-vs-stock (8.2e-4 and 3.1e-6).

The one intended change is the causal-conv1d boundary fix. At seqlen 2049, stock's `dx` is off by
36% of its max at the chunk boundary, `dweight` by 2.5% and `dbias` by 1.6%; the patched results
match the `F.conv1d` reference. At other lengths conv outputs are unchanged.

**2. Deterministic mode only changes rounding.** Kernel and module outputs and gradients, compared
to stock ([`kernel_report.txt`](tests/equivalence/results/kernel_report.txt)), max relative
difference:

| | deterministic vs stock | stock vs stock (its own run-to-run spread) |
|---|---|---|
| Mamba1 scan / `mamba_inner_fn` / module (fp32) | ≤ 3.2e-7 | ≤ 3.0e-7 |
| Mamba2 SSD kernel (fp32) / module | ≤ 2.2e-6 / ≤ 7e-7 | ≤ 2.4e-7 |
| Mamba3 SISO kernel / module | ≤ 1.5e-5, except atomic `dK` 3.4e-3 / ≤ 8.6e-5 | 0, except atomic `dK` 6.8e-3 / 0 |

Mamba2 and Mamba3 differ more than Mamba1 because deterministic mode runs fixed kernel configs,
which use a different reduction order. Mamba3's kernels also use fast sin/cos/tanh
approximations. Against a float64 reference, deterministic mode is as accurate as default (see
`tests/test_mamba2_mamba3.py`).

300-step `MixerModel` training on a learnable synthetic language (loss 4.3 → 1.44), with autotuning
left on as a user would run it ([`curve_report.txt`](tests/equivalence/results/curve_report.txt)).
The table shows the max loss gap at any step vs stock run 1:

| | stock run 2 | patched default | patched deterministic | final loss (all runs) |
|---|---|---|---|---|
| Mamba1 | 2.4e-7 | 4.8e-7 | 2.4e-7 | 1.4462 |
| Mamba2 | 2.4e-7 | 2.4e-7 | 2.4e-7 | 1.4479 |
| Mamba3 | 6.0e-4 | 1.0e-3 | 8.8e-4 | 1.4354 – 1.4359 |

Stock Mamba3 already diverges from itself at step 0, because autotune picks different forward
configs in each process. The patched runs' larger gap traces to the same cause: section 1 shows
that with configs pinned, patched default matches stock bit for bit.

Rerun: `tests/equivalence/run_all.sh STOCK_PY PATCHED_PY OUTDIR` (kernel report + training
curves), then `pinned_compare.py` for the pinned-config test (see the script headers).

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
tests/equivalence/        stock-vs-patched equivalence harness + results/
```

The git history also holds the earlier standalone fork for mamba_ssm 2.2.4 (Mamba1 only).
