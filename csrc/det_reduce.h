/******************************************************************************
 * Deterministic fixed-order reductions used to combine the per-block partial
 * gradients written by the deterministic backward kernels.
 ******************************************************************************/

#pragma once

#include <ATen/ATen.h>

// out[o, i] (+)= sum_{n=0}^{N-1} in[o, n, i], summed sequentially in increasing n (fp32).
//   in  : float32, element (o, n, i) at in_ptr[o * in_outer_stride + n * in_n_stride + i]
//   out : float32, element (o, i)    at out_ptr[o * out_outer_stride + i]
// If accumulate is true the running sum starts from the current value of out, so a sequence of
// calls over consecutive slices of n produces the same result on every run.
void det_sum_middle(const float *in_ptr, float *out_ptr,
                    int64_t outer, int64_t N, int64_t inner,
                    int64_t in_outer_stride, int64_t in_n_stride, int64_t out_outer_stride,
                    bool accumulate);

// Convenience wrapper: reduce a contiguous float32 tensor over dimension 0. Returns float32.
at::Tensor det_sum_dim0(const at::Tensor &in);
