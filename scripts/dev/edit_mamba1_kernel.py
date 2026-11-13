"""One-off edit script used to author patches/mamba/0001 (kept for re-porting to new upstream tags).
Run from the root of an upstream mamba checkout. Every anchor is asserted, so it fails loudly if
upstream changed the code it touches."""
import re

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

# ---------------------------------------------------------------- params
p = "csrc/selective_scan/selective_scan.h"
s = open(p).read()
s = sub(s, """struct SSMParamsBwd: public SSMParamsBase {
""", """struct SSMParamsBwd: public SSMParamsBase {
    // Deterministic mode: instead of atomically accumulating dA, dB, dC, dD and ddelta_bias, every
    // (batch, channel) block stores its partial result into its own slot of a float32 workspace,
    // which the host then reduces in a fixed order (see selective_scan_det_reduce.cu). The kernel
    // may be launched over a slice of channels [dim_offset, dim_offset + n_local_dims) so that the
    // per-channel dB/dC workspace stays within a memory budget.
    bool deterministic;
    int dim_offset;
    int n_local_dims;
""")
open(p, "w").write(s)

# ---------------------------------------------------------------- kernel
p = "csrc/selective_scan/selective_scan_bwd_kernel.cuh"
s = open(p).read()
s = sub(s, """template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);""",
"""// Accumulate into a gradient: atomically by default, or a plain store into this block's own
// workspace slot in deterministic mode (the host reduces the workspace in a fixed order).
template<typename T>
__device__ __forceinline__ void det_or_atomic_add(bool deterministic, T *ptr, T val) {
    if (deterministic) { *ptr = val; } else { gpuAtomicAdd(ptr, val); }
}

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);""")
s = sub(s, """    const int dim_id = blockIdx.y;
""", """    const int local_dim_id = blockIdx.y;
    const int dim_id = blockIdx.y + params.dim_offset;
    const bool det = params.deterministic;
    // Index of this block's slot in the (batch, dim) deterministic workspaces.
    const size_t bd_idx = size_t(batch_id) * params.dim + dim_id;
""")
s = sub(s, """    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (!kIsVariableB ? dim_id * params.dB_d_stride : batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (!kIsVariableC ? dim_id * params.dC_d_stride : batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
""", """    // Deterministic workspaces (dstate strides are set to 1 by the host for the (batch, dim, dstate) ones):
    //   dA, constant dB / dC : weight_t (batch, dim, dstate)
    //   variable dB / dC     : float    (batch, n_local_dims, dstate, seqlen [* 2 if complex])
    //   dD, ddelta_bias      : float    (batch, dim)
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr)
        + (det ? bd_idx * params.dstate : size_t(dim_id) * params.dA_d_stride);
    weight_t *dB = !det
        ? reinterpret_cast<weight_t *>(params.dB_ptr)
            + (!kIsVariableB ? dim_id * params.dB_d_stride : batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) + group_id * params.dB_group_stride)
        : (!kIsVariableB
            ? reinterpret_cast<weight_t *>(params.dB_ptr) + bd_idx * params.dstate
            : reinterpret_cast<weight_t *>(reinterpret_cast<float *>(params.dB_ptr) + size_t(batch_id) * params.dB_batch_stride + size_t(local_dim_id) * params.dB_group_stride));
    weight_t *dC = !det
        ? reinterpret_cast<weight_t *>(params.dC_ptr)
            + (!kIsVariableC ? dim_id * params.dC_d_stride : batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) + group_id * params.dC_group_stride)
        : (!kIsVariableC
            ? reinterpret_cast<weight_t *>(params.dC_ptr) + bd_idx * params.dstate
            : reinterpret_cast<weight_t *>(reinterpret_cast<float *>(params.dC_ptr) + size_t(batch_id) * params.dC_batch_stride + size_t(local_dim_id) * params.dC_group_stride));
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + (det ? bd_idx : size_t(dim_id));
""")
s = sub(s, """    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;""",
"""    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + (det ? bd_idx : size_t(dim_id));""")
# every remaining atomic -> det_or_atomic_add
s = sub(s, "gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]);", "det_or_atomic_add(det, dB_cur + i * kNThreads, dB_vals[i]);")
s = sub(s, "gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]);", "det_or_atomic_add(det, dC_cur + i * kNThreads, dC_vals[i]);")
s = sub(s, "gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals_f[i]);", "det_or_atomic_add(det, dB_cur + i * kNThreads, dB_vals_f[i]);")
s = sub(s, "gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals_f[i]);", "det_or_atomic_add(det, dC_cur + i * kNThreads, dC_vals_f[i]);")
s = sub(s, "gpuAtomicAdd(dD, dD_val);", "det_or_atomic_add(det, dD, dD_val);")
s = sub(s, "gpuAtomicAdd(ddelta_bias, ddelta_bias_val);", "det_or_atomic_add(det, ddelta_bias, ddelta_bias_val);")
s = sub(s, "gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);",
        "det_or_atomic_add<weight_t>(det, &(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);")
s = sub(s, """            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * conj(C[state_idx * params.C_dstate_stride]) : dBC_val);""",
"""            det_or_atomic_add<weight_t>(det, &(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * conj(C[state_idx * params.C_dstate_stride]) : dBC_val);""")
s = sub(s, """            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                        !kIsVariableB ? dBC_val * conj(B[state_idx * params.B_dstate_stride]) : dBC_val);""",
"""            det_or_atomic_add<weight_t>(det, &(dC[state_idx * params.dC_dstate_stride]),
                        !kIsVariableB ? dBC_val * conj(B[state_idx * params.B_dstate_stride]) : dBC_val);""")
assert s.count("gpuAtomicAdd(") == 1, "an atomic was left unconverted"  # only the one inside det_or_atomic_add
s = sub(s, "dim3 grid(params.batch, params.dim);", "dim3 grid(params.batch, params.n_local_dims);")
open(p, "w").write(s)
print("kernel edited")
