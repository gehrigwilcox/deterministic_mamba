# deterministic_mamba

A bitwise-deterministic fork of the **Mamba1** kernels and `MixerModel` from
[`mamba_ssm` 2.2.4](https://github.com/state-spaces/mamba/tree/v2.2.4) (and the
[`causal-conv1d` 1.5.0.post8](https://github.com/Dao-AILab/causal-conv1d) kernels it uses).

Same constructor arguments, same `state_dict` keys, same math. The difference is that two
training runs with the same seed and data produce **bit-identical** gradients and weights on the
same GPU and software stack.

```python
from deterministic_mamba import MixerModel, MambaLMHeadModel, Mamba, enable_full_determinism

enable_full_determinism(seed=0)  # torch.use_deterministic_algorithms + cuBLAS/cuDNN settings
model = MixerModel(d_model=768, n_layer=24, d_intermediate=0, vocab_size=50277,
                   rms_norm=True, fused_add_norm=True, residual_in_fp32=True, device="cuda")
```

To switch an existing codebase, change the imports:

| mamba_ssm 2.2.4 | deterministic_mamba |
|---|---|
| `mamba_ssm.models.mixer_seq_simple.MixerModel` | `deterministic_mamba.MixerModel` |
| `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel` | `deterministic_mamba.MambaLMHeadModel` |
| `mamba_ssm.Mamba` | `deterministic_mamba.Mamba` |
| `mamba_ssm.ops.selective_scan_interface.{selective_scan_fn, mamba_inner_fn}` | `deterministic_mamba.{selective_scan_fn, mamba_inner_fn}` |
| `causal_conv1d.causal_conv1d_fn` | `deterministic_mamba.causal_conv1d_fn` |

Checkpoints load in both directions without conversion.

## Where the non-determinism came from, and the fix

| Source (upstream 2.2.4) | Why it is non-deterministic | Fix |
|---|---|---|
| `selective_scan_bwd_kernel`: `dB`, `dC` (input-dependent B/C) | Every channel of a group `atomicAdd`s into the same `dB[b, g, n, l]`, so float summation order follows block scheduling | Each `(batch, channel)` block **stores** its contribution in its own workspace row. A fixed-order kernel (`csrc/det_reduce.cu`) sums the rows in channel order. |
| same kernel: `dA`, `dD`, `ddelta_bias`, and constant `dB`/`dC` | `atomicAdd` across the batch | Per-`(batch, channel)` workspace, then a fixed-order sum over the batch |
| `causal_conv1d_bwd_kernel`: `dweight`, `dbias` | `atomicAdd` across the batch | Per-`(batch, channel)` workspace, then a fixed-order sum over the batch |
| Triton `layer_norm.py` (RMSNorm / fused add-norm) | `@triton.autotune` over `num_warps` can choose a different config in each process, which changes the `tl.sum` reduction order | Autotune removed; `num_warps` is a fixed function of the block size |
| Everything else (GEMMs, embedding, loss) | Controlled by PyTorch's global flags | `enable_full_determinism()` sets them, including `CUBLAS_WORKSPACE_CONFIG` |

The sliced workspace (next section) adds rows in the same sequence whatever the slice size, so
the result is also bitwise independent of the memory budget. A test checks this.

### Memory budget for dB/dC

Writing per-channel `dB`/`dC` contributions needs `batch × channels × d_state × seqlen × 4 B`.
The host processes channels in slices that fit a budget (default **256 MiB**):

```python
deterministic_mamba.set_workspace_mb(64)   # or: export DETERMINISTIC_MAMBA_WORKSPACE_MB=64
```

## Measured (GTX 1660 SUPER, torch 2.6, CUDA 12.6)

Ten backward calls on identical inputs (batch 4, d_inner 512, L 1500), counting how many
different results each gradient took:

| | du | ddelta | dz | dA | dB | dC | dD | ddelta_bias |
|---|---|---|---|---|---|---|---|---|
| upstream 2.2.4 | 1 | 1 | 1 | **10** | **10** | **10** | **10** | **10** |
| deterministic_mamba | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

`du`, `ddelta` and `dz` are bitwise equal to upstream. The rest agree with upstream to about 1e-7
relative in fp32 (only the summation order differs).

Selective-scan backward cost (batch 8, d_inner 1536, L 2048, d_state 16, bf16):

| | time | extra peak memory |
|---|---|---|
| upstream 2.2.4 | 28.8 ms | 147 MiB |
| deterministic, 64 MiB budget | 43.0 ms | 212 MiB |
| deterministic, 256 MiB budget (default) | 40.7 ms | 404 MiB |

This is the cost of the scan backward only. The rest of a training step (GEMMs, norms,
conv) runs at the same speed as upstream.

## Bug fix vs upstream causal-conv1d

`causal_conv1d_bwd_kernel` with SiLU (the path `mamba_inner_fn` always uses) bounds-checks the
wrong index when it loads the last `kNElts` inputs of the previous chunk. If the final chunk holds
fewer than `kNElts` valid elements (for example seqlen 2049, 4097, …), those inputs are read as 0.
That gives wrong `dx`, `dweight` and `dbias` at that chunk boundary. The fork fixes the bounds
check (see `csrc/causal_conv1d/causal_conv1d_bwd.cu`), and `tests/test_causal_conv1d.py` covers
it.

## Scope / not supported

* **Mamba1 only.** `ssm_cfg={"layer": "Mamba2"}` and attention layers (`attn_layer_idx`) raise
  `NotImplementedError`. Mamba2's Triton SSD kernels have their own atomics and autotuning and
  would need a separate fork.
* Real-valued `A` only (complex selective scan removed). Mamba1 never uses complex `A`.
* `causal_conv1d_fn`: channel-first layout only; no `seq_idx` / `initial_states` /
  `final_states` (Mamba2 features).
* Determinism holds for a fixed GPU model and software stack (driver, CUDA, torch, triton).
  Different GPUs (different SM counts, for example) can give different, equally valid bits.
* Inference: `step()` decoding uses the pure-PyTorch conv-state update instead of the
  `causal_conv1d_update` CUDA kernel. The forward pass is otherwise unchanged.

## Build

The build needs a CUDA toolkit matching your torch build (`nvcc` on `PATH` / `CUDA_HOME`):

```bash
pip install torch einops triton packaging
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" pip install --no-build-isolation -e .
```

Without `TORCH_CUDA_ARCH_LIST` it builds sm_75/80/86/89/90. For a conda CUDA toolkit, also export
`CPATH=$CONDA_PREFIX/targets/x86_64-linux/include`.

## Tests

```bash
pytest tests
```

* `test_selective_scan.py`: forward and all gradients against the pure-PyTorch reference
  (fp32/fp16/bf16, 1 and >1 chunks, variable/constant/grouped B and C, with and without z).
  Also bitwise repeatability, and bitwise independence from the workspace budget.
* `test_causal_conv1d.py`: against `F.conv1d`, plus repeatability.
* `test_mixer_model.py`: AdamW training of a `MixerModel` is bitwise reproducible in-process and
  across two separate processes. The fast and slow paths agree, the model runs under
  `torch.use_deterministic_algorithms(True)`, and recurrent decoding matches the parallel scan.

## Layout

```
csrc/selective_scan/   forked selective-scan kernels (bwd kernel + host code changed)
csrc/causal_conv1d/    forked causal conv1d kernels (channel-first, bwd changed, bug fix)
csrc/det_reduce.cu     fixed-order reductions
deterministic_mamba/   Python package (ops, modules, models) mirroring mamba_ssm's layout
```

The first commit vendors the unmodified upstream sources, so `git diff <first commit>` shows every
change made for determinism.

Licenses: Apache-2.0 (mamba, see `LICENSE.mamba`) and BSD-3-Clause (causal-conv1d, see
`LICENSE.causal_conv1d`).
