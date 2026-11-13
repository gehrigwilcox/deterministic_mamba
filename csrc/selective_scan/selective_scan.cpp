/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 *
 * Deterministic fork (based on mamba_ssm 2.2.4): the backward pass writes
 * per-(batch, channel) partial gradients into float32 workspaces and reduces
 * them with fixed-order kernels (det_reduce.cu) instead of atomicAdd.
 ******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <vector>

#include "selective_scan.h"
#include "../det_reduce.h"

#include <algorithm>

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

// Deterministic fork: only real (float32) A is supported.
#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float) {                                            \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        const at::Tensor z,
                        const at::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

void set_ssm_params_bwd(SSMParamsBwd &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor z,
                        const at::Tensor out,
                        const at::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        const at::Tensor dout,
                        const at::Tensor du,
                        const at::Tensor ddelta,
                        const at::Tensor dA,
                        const at::Tensor dB,
                        const at::Tensor dC,
                        const at::Tensor dz,
                        void* dD_ptr,
                        void* ddelta_bias_ptr,
                        bool has_z,
                        bool delta_softplus,
                        bool recompute_out_z) {
    // Pass in "dout" instead of "out", we're not gonna use "out" unless we have z
    set_ssm_params_fwd(params, batch, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, has_z ? out : dout,
                       has_z ? z : dout,
                       // If not recompute_out_z, pass dout instead of out_z.
                       // This won't be used by the bwd kernel
                       recompute_out_z ? out_z : dout,
                       D_ptr, delta_bias_ptr, x_ptr, has_z, delta_softplus);
    if (!recompute_out_z) { params.out_z_ptr = nullptr; }

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    params.dz_ptr = has_z ? dz.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);
    if (!is_variable_B) {
        params.dB_d_stride = dB.stride(0);
    } else {
        params.dB_batch_stride = dB.stride(0);
        params.dB_group_stride = dB.stride(1);
    }
    params.dB_dstate_stride = !is_variable_B ? dB.stride(1) : dB.stride(2);
    if (!is_variable_C) {
        params.dC_d_stride = dC.stride(0);
    } else {
        params.dC_batch_stride = dC.stride(0);
        params.dC_group_stride = dC.stride(1);
    }
    params.dC_dstate_stride = !is_variable_C ? dC.stride(1) : dC.stride(2);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);
    if (has_z) {
        params.dz_batch_stride = dz.stride(0);
        params.dz_d_stride = dz.stride(1);
    }
}

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  bool delta_softplus) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float,
                "deterministic_mamba: selective_scan only supports real float32 A (complex is not supported)");

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = torch::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::empty_like(delta);
    at::Tensor x;
    x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(weight_type));

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x.data_ptr(),
                       has_z,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<at::Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const c10::optional<at::Tensor> &out_,
                  c10::optional<at::Tensor> &dz_,
                  bool delta_softplus,
                  bool recompute_out_z,
                  int64_t workspace_mb) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float,
                "deterministic_mamba: selective_scan only supports real float32 A (complex is not supported)");

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    TORCH_CHECK(dout.scalar_type() == input_type);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    at::Tensor z, out, dz, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);

        TORCH_CHECK(out_.has_value());
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == input_type);
        TORCH_CHECK(out.is_cuda());
        TORCH_CHECK(out.stride(-1) == 1 || out.size(-1) == 1);
        CHECK_SHAPE(out, batch_size, dim, seqlen);

        if (dz_.has_value()) {
            dz = dz_.value();
            TORCH_CHECK(dz.scalar_type() == input_type);
            TORCH_CHECK(dz.is_cuda());
            TORCH_CHECK(dz.stride(-1) == 1 || dz.size(-1) == 1);
            CHECK_SHAPE(dz, batch_size, dim, seqlen);
        } else {
            dz = torch::empty_like(z);
        }
        if (recompute_out_z) {
            out_z = torch::empty_like(out);
        }
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    if (n_chunks > 1) { TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);

    // ---- Deterministic workspaces (float32, fully overwritten by the kernel) ----
    auto fopts = u.options().dtype(torch::kFloat32);
    at::Tensor dA_ws = torch::empty({batch_size, dim, dstate}, fopts);
    at::Tensor dB_ws = !is_variable_B ? torch::empty({batch_size, dim, dstate}, fopts) : at::Tensor();
    at::Tensor dC_ws = !is_variable_C ? torch::empty({batch_size, dim, dstate}, fopts) : at::Tensor();
    at::Tensor dD_ws, ddelta_bias_ws;
    if (D_.has_value()) { dD_ws = torch::empty({batch_size, dim}, fopts); }
    if (delta_bias_.has_value()) { ddelta_bias_ws = torch::empty({batch_size, dim}, fopts); }

    // Final float32 results for variable B / C: (batch, n_groups, dstate, seqlen).
    at::Tensor dB_var = is_variable_B ? torch::empty({batch_size, n_groups, dstate, seqlen}, fopts) : at::Tensor();
    at::Tensor dC_var = is_variable_C ? torch::empty({batch_size, n_groups, dstate, seqlen}, fopts) : at::Tensor();

    // Channels of one group all contribute to the same dB/dC[b, g]. Each channel writes its own
    // workspace row; channels are processed in slices so the workspace stays within the budget.
    const int ratio = dim / n_groups;
    const int n_var = int(is_variable_B) + int(is_variable_C);
    int slice = ratio;
    if (n_var > 0) {
        const int64_t per_dim_bytes = int64_t(batch_size) * dstate * seqlen * 4 * n_var;
        const int64_t budget = std::max<int64_t>(workspace_mb, 1) * 1024 * 1024;
        slice = int(std::min<int64_t>(ratio, std::max<int64_t>(1, budget / per_dim_bytes)));
    }
    at::Tensor dB_slice_ws = is_variable_B ? torch::empty({batch_size, slice, dstate, seqlen}, fopts) : at::Tensor();
    at::Tensor dC_slice_ws = is_variable_C ? torch::empty({batch_size, slice, dstate, seqlen}, fopts) : at::Tensor();

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, z, out, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x_.has_value() ? x_.value().data_ptr() : nullptr,
                       dout, du, ddelta,
                       dA_ws,
                       is_variable_B ? dB_slice_ws : dB_ws,
                       is_variable_C ? dC_slice_ws : dC_ws,
                       dz,
                       D_.has_value() ? dD_ws.data_ptr() : nullptr,
                       delta_bias_.has_value() ? ddelta_bias_ws.data_ptr() : nullptr,
                       has_z, delta_softplus, recompute_out_z);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t bc_inner = int64_t(dstate) * seqlen;
    for (int g = 0; g < n_groups; ++g) {
        // With constant B and C there is no per-group workspace: one launch covers all channels.
        const int g_begin = n_var > 0 ? g * ratio : 0;
        const int g_end = n_var > 0 ? (g + 1) * ratio : dim;
        for (int d0 = g_begin; d0 < g_end; d0 += slice) {
            const int n = std::min(slice, g_end - d0);
            params.dim_offset = d0;
            params.n_local_dims = n;
            DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_bwd", [&] {
                DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_bwd", [&] {
                    selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
                });
            });
            // Fold this slice of channels into dB/dC[:, g] in increasing channel order.
            const bool accumulate = d0 != g_begin;
            if (is_variable_B) {
                det_sum_middle(dB_slice_ws.data_ptr<float>(), dB_var.data_ptr<float>() + g * bc_inner,
                               batch_size, n, bc_inner, dB_slice_ws.stride(0), dB_slice_ws.stride(1),
                               dB_var.stride(0), accumulate);
            }
            if (is_variable_C) {
                det_sum_middle(dC_slice_ws.data_ptr<float>(), dC_var.data_ptr<float>() + g * bc_inner,
                               batch_size, n, bc_inner, dC_slice_ws.stride(0), dC_slice_ws.stride(1),
                               dC_var.stride(0), accumulate);
            }
        }
        if (n_var == 0) { break; }
    }

    // Reduce the per-batch partials over the batch dimension in a fixed order.
    at::Tensor dA = det_sum_dim0(dA_ws);
    at::Tensor dB = is_variable_B ? dB_var.to(B.dtype()) : det_sum_dim0(dB_ws).to(B.dtype());
    at::Tensor dC = is_variable_C ? dC_var.to(C.dtype()) : det_sum_dim0(dC_ws).to(C.dtype());
    at::Tensor dD, ddelta_bias;
    if (D_.has_value()) { dD = det_sum_dim0(dD_ws); }
    if (delta_bias_.has_value()) { ddelta_bias = det_sum_dim0(ddelta_bias_ws); }

    std::vector<at::Tensor> result = {du, ddelta, dA, dB, dC, dD, ddelta_bias};
    if (has_z) { result.push_back(dz); }
    if (recompute_out_z) { result.push_back(out_z); }
    return result;
}
