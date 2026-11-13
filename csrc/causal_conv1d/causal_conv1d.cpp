/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 *
 * Deterministic fork (based on causal-conv1d 1.5.0.post8). Channel-first layout
 * only (the layout Mamba1's mamba_inner_fn uses). The backward pass writes
 * per-(batch, channel) partial dweight/dbias into float32 workspaces that are
 * reduced over the batch in a fixed order.
 ******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>

#include "causal_conv1d.h"
#include "../det_reduce.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream);

static void set_conv_params_fwd(ConvParamsBase &params,
                                const size_t batch, const size_t dim, const size_t seqlen, const size_t width,
                                const at::Tensor x, const at::Tensor weight, const at::Tensor out,
                                void* bias_ptr, bool silu_activation) {
    memset(&params, 0, sizeof(params));
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = silu_activation;
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(-1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(-1);
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
}

static void check_common(const at::Tensor &x, const at::Tensor &weight, const c10::optional<at::Tensor> &bias_) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());
    TORCH_CHECK(x.dim() == 3);
    const int batch_size = x.size(0), dim = x.size(1), seqlen = x.size(2);
    const int width = weight.size(-1);
    CHECK_SHAPE(weight, dim, width);
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");
    TORCH_CHECK(x.stride(2) == 1,
                "deterministic_mamba causal_conv1d only supports channel-first (stride(-1) == 1) inputs");
    (void)batch_size; (void)seqlen;
    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }
}

at::Tensor
causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  bool silu_activation) {
    check_common(x, weight, bias_);
    const int batch_size = x.size(0), dim = x.size(1), seqlen = x.size(2);
    const int width = weight.size(-1);

    at::Tensor out = torch::empty_like(x, at::MemoryFormat::Contiguous);
    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr, silu_activation);

    at::cuda::CUDAGuard device_guard{x.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_fwd", [&] {
            causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    return out;
}

std::vector<at::Tensor>
causal_conv1d_bwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  at::Tensor &dout,
                  c10::optional<at::Tensor> &dx_,
                  bool silu_activation) {
    check_common(x, weight, bias_);
    TORCH_CHECK(dout.is_cuda());
    const int batch_size = x.size(0), dim = x.size(1), seqlen = x.size(2);
    const int width = weight.size(-1);
    CHECK_SHAPE(dout, batch_size, dim, seqlen);
    TORCH_CHECK(dout.scalar_type() == x.scalar_type());
    if (dout.stride(2) != 1) { dout = dout.contiguous(); }

    at::Tensor dx;
    if (dx_.has_value()) {
        dx = dx_.value();
        TORCH_CHECK(dx.scalar_type() == x.scalar_type());
        TORCH_CHECK(dx.is_cuda());
        CHECK_SHAPE(dx, batch_size, dim, seqlen);
        TORCH_CHECK(dx.stride(2) == 1);
    } else {
        dx = torch::empty_like(x, at::MemoryFormat::Contiguous);
    }

    at::cuda::CUDAGuard device_guard{x.device()};

    auto fopts = x.options().dtype(at::kFloat);
    at::Tensor dweight_ws = torch::empty({batch_size, dim, width}, fopts);
    at::Tensor dbias_ws;
    if (bias_.has_value()) { dbias_ws = torch::empty({batch_size, dim}, fopts); }

    ConvParamsBwd params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, dout,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr, silu_activation);
    params.dout_ptr = dout.data_ptr();
    params.dx_ptr = dx.data_ptr();
    params.dweight_ptr = dweight_ws.data_ptr();
    params.dbias_ptr = bias_.has_value() ? dbias_ws.data_ptr() : nullptr;
    params.dout_batch_stride = dout.stride(0);
    params.dout_c_stride = dout.stride(1);
    params.dout_l_stride = dout.stride(2);
    params.dweight_c_stride = width;
    params.dweight_width_stride = 1;
    params.dx_batch_stride = dx.stride(0);
    params.dx_c_stride = dx.stride(1);
    params.dx_l_stride = dx.stride(2);
    params.dinitial_states_ptr = nullptr;
    params.dfinal_states_ptr = nullptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(weight.scalar_type(), "causal_conv1d_bwd", [&] {
            causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
        });
    });

    at::Tensor dweight = det_sum_dim0(dweight_ws).to(weight.dtype());
    at::Tensor dbias;
    if (bias_.has_value()) { dbias = det_sum_dim0(dbias_ws).to(bias_.value().dtype()); }
    return {dx, dweight, dbias};
}
