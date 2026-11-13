/******************************************************************************
 * Deterministic fixed-order reductions. One thread owns one output element and
 * adds its inputs in a fixed sequential order, so the floating point result
 * depends only on the data and never on scheduling.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "det_reduce.h"

__global__ void det_sum_middle_kernel(const float *__restrict__ in, float *__restrict__ out,
                                      int64_t outer, int64_t N, int64_t inner,
                                      int64_t in_outer_stride, int64_t in_n_stride,
                                      int64_t out_outer_stride, bool accumulate) {
    const int64_t total = outer * inner;
    for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += int64_t(gridDim.x) * blockDim.x) {
        const int64_t o = idx / inner;
        const int64_t i = idx - o * inner;
        const float *src = in + o * in_outer_stride + i;
        float *dst = out + o * out_outer_stride + i;
        float acc = accumulate ? *dst : 0.f;
        for (int64_t n = 0; n < N; ++n) { acc += src[n * in_n_stride]; }
        *dst = acc;
    }
}

void det_sum_middle(const float *in_ptr, float *out_ptr,
                    int64_t outer, int64_t N, int64_t inner,
                    int64_t in_outer_stride, int64_t in_n_stride, int64_t out_outer_stride,
                    bool accumulate) {
    const int64_t total = outer * inner;
    if (total == 0) { return; }
    constexpr int kThreads = 256;
    const int64_t max_blocks = 65535;
    const int64_t blocks = std::min<int64_t>((total + kThreads - 1) / kThreads, max_blocks);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    det_sum_middle_kernel<<<blocks, kThreads, 0, stream>>>(
        in_ptr, out_ptr, outer, N, inner, in_outer_stride, in_n_stride, out_outer_stride, accumulate);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor det_sum_dim0(const at::Tensor &in) {
    TORCH_CHECK(in.scalar_type() == at::kFloat && in.is_contiguous() && in.dim() >= 1);
    auto out_sizes = in.sizes().vec();
    const int64_t N = out_sizes[0];
    out_sizes.erase(out_sizes.begin());
    at::Tensor out = at::empty(out_sizes, in.options());
    const int64_t inner = out.numel();
    det_sum_middle(in.data_ptr<float>(), out.data_ptr<float>(), 1, N, inner, 0, inner, 0, false);
    return out;
}
