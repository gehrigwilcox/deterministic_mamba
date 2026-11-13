"""One-off edit script used to author patches/mamba/0001 (host side). Run from an upstream mamba checkout."""

def sub(s, old, new, count=1):
    n = s.count(old)
    assert n == count, f"expected {count} occurrence(s), found {n}:\n{old}"
    return s.replace(old, new)

p = "csrc/selective_scan/selective_scan.cpp"
s = open(p).read()
s = sub(s, '#include "selective_scan.h"\n', '''#include "selective_scan.h"
#include "selective_scan_det_reduce.h"

#include <algorithm>
#include <cstdlib>

// Default for callers that don't pass `deterministic`: MAMBA_DETERMINISTIC=1/0 overrides,
// otherwise follow torch.use_deterministic_algorithms(...).
static bool default_deterministic_mode() {
    const char *env = std::getenv("MAMBA_DETERMINISTIC");
    if (env != nullptr && env[0] != '\\0') { return env[0] == '1'; }
    return at::globalContext().deterministicAlgorithms();
}
''')
s = sub(s, """                  bool delta_softplus,
                  bool recompute_out_z) {""", """                  bool delta_softplus,
                  bool recompute_out_z,
                  const c10::optional<bool> &deterministic_,
                  int64_t workspace_mb) {""")

a = s.index("    at::Tensor du = torch::empty_like(u);")
b = s.index("    std::vector<at::Tensor> result = {du, ddelta, dA, dB.to(B.dtype()), dC.to(C.dtype()), dD, ddelta_bias};")
old_tail = s[a:b]
assert "selective_scan_bwd_cuda<input_t, weight_t>(params, stream);" in old_tail
new_tail = r'''    const bool deterministic = deterministic_.has_value() ? deterministic_.value() : default_deterministic_mode();

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA, dB, dC, dD, ddelta_bias;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (!deterministic) {
        // ---------------- Upstream path: atomic accumulation ----------------
        dA = torch::zeros_like(A);
        dB = !is_variable_B ? torch::zeros_like(B) : torch::zeros_like(B, B.options().dtype(torch::kFloat32));
        dC = !is_variable_C ? torch::zeros_like(C) : torch::zeros_like(C, C.options().dtype(torch::kFloat32));
        if (D_.has_value()) { dD = torch::zeros_like(D_.value()); }
        if (delta_bias_.has_value()) { ddelta_bias = torch::zeros_like(delta_bias_.value()); }

        SSMParamsBwd params;
        set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                           u, delta, A, B, C, z, out, out_z,
                           D_.has_value() ? D_.value().data_ptr() : nullptr,
                           delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                           x_.has_value() ? x_.value().data_ptr() : nullptr,
                           dout, du, ddelta, dA, dB, dC, dz,
                           D_.has_value() ? dD.data_ptr() : nullptr,
                           delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
                           has_z, delta_softplus, recompute_out_z);
        params.deterministic = false;
        params.dim_offset = 0;
        params.n_local_dims = dim;
        DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_bwd", [&] {
            DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_bwd", [&] {
                selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
            });
        });
    } else {
        // ---------------- Deterministic path ----------------
        // Every (batch, channel) block writes its partial gradients into its own workspace slot
        // (no atomics); they are then reduced with fixed-order kernels.
        auto fopts = u.options().dtype(torch::kFloat32);
        auto wopts = A.options();  // float32 or complex64
        at::Tensor dA_ws = torch::empty({batch_size, dim, dstate}, wopts);
        at::Tensor dB_ws = !is_variable_B ? torch::empty({batch_size, dim, dstate}, wopts) : at::Tensor();
        at::Tensor dC_ws = !is_variable_C ? torch::empty({batch_size, dim, dstate}, wopts) : at::Tensor();
        at::Tensor dD_ws, ddelta_bias_ws;
        if (D_.has_value()) { dD_ws = torch::empty({batch_size, dim}, fopts); }
        if (delta_bias_.has_value()) { ddelta_bias_ws = torch::empty({batch_size, dim}, fopts); }

        // Variable B / C: all channels of a group contribute to the same dB/dC[b, g]. Each channel
        // writes its own row; channels are processed in slices so the workspace fits the budget.
        const int64_t L_f = int64_t(seqlen) * (is_complex ? 2 : 1);  // float elements per row
        at::Tensor dB_var = is_variable_B ? torch::empty({batch_size, n_groups, dstate, L_f}, fopts) : at::Tensor();
        at::Tensor dC_var = is_variable_C ? torch::empty({batch_size, n_groups, dstate, L_f}, fopts) : at::Tensor();
        const int ratio = dim / n_groups;
        const int n_var = int(is_variable_B) + int(is_variable_C);
        int slice = n_var > 0 ? ratio : dim;
        if (n_var > 0) {
            const int64_t per_dim_bytes = int64_t(batch_size) * dstate * L_f * 4 * n_var;
            const int64_t budget = std::max<int64_t>(workspace_mb, 1) * 1024 * 1024;
            slice = int(std::min<int64_t>(ratio, std::max<int64_t>(1, budget / per_dim_bytes)));
        }
        at::Tensor dB_slice = is_variable_B ? torch::empty({batch_size, slice, dstate, L_f}, fopts) : at::Tensor();
        at::Tensor dC_slice = is_variable_C ? torch::empty({batch_size, slice, dstate, L_f}, fopts) : at::Tensor();

        SSMParamsBwd params;
        set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                           u, delta, A, B, C, z, out, out_z,
                           D_.has_value() ? D_.value().data_ptr() : nullptr,
                           delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                           x_.has_value() ? x_.value().data_ptr() : nullptr,
                           dout, du, ddelta, dA_ws,
                           is_variable_B ? dB_slice : dB_ws,
                           is_variable_C ? dC_slice : dC_ws,
                           dz,
                           D_.has_value() ? dD_ws.data_ptr() : nullptr,
                           delta_bias_.has_value() ? ddelta_bias_ws.data_ptr() : nullptr,
                           has_z, delta_softplus, recompute_out_z);
        params.deterministic = true;
        // Workspace layouts (see selective_scan_bwd_kernel.cuh): the (batch, dim, dstate) ones are
        // indexed directly by the kernel with unit dstate stride.
        params.dA_dstate_stride = 1;
        if (!is_variable_B) { params.dB_dstate_stride = 1; }
        if (!is_variable_C) { params.dC_dstate_stride = 1; }

        const int64_t bc_inner = int64_t(dstate) * L_f;
        for (int g = 0; g < (n_var > 0 ? n_groups : 1); ++g) {
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
                // Fold this slice into dB/dC[:, g] in increasing channel order. The running sum
                // continues across slices, so the bits don't depend on the slice size.
                const bool accumulate = d0 != g_begin;
                if (is_variable_B) {
                    det_sum_middle(dB_slice.data_ptr<float>(), dB_var.data_ptr<float>() + g * bc_inner,
                                   batch_size, n, bc_inner, dB_slice.stride(0), dB_slice.stride(1),
                                   dB_var.stride(0), accumulate);
                }
                if (is_variable_C) {
                    det_sum_middle(dC_slice.data_ptr<float>(), dC_var.data_ptr<float>() + g * bc_inner,
                                   batch_size, n, bc_inner, dC_slice.stride(0), dC_slice.stride(1),
                                   dC_var.stride(0), accumulate);
                }
            }
        }
        // Reduce the per-batch partials over the batch dimension in a fixed order.
        dA = det_sum_dim0(dA_ws);
        dB = is_variable_B ? dB_var : det_sum_dim0(dB_ws);
        dC = is_variable_C ? dC_var : det_sum_dim0(dC_ws);
        if (D_.has_value()) { dD = det_sum_dim0(dD_ws); }
        if (delta_bias_.has_value()) { ddelta_bias = det_sum_dim0(ddelta_bias_ws); }
    }
'''
s = s[:a] + new_tail + s[b:]
s = sub(s, '''    m.def("bwd", &selective_scan_bwd, "Selective scan backward");''',
'''    m.def("bwd", &selective_scan_bwd, "Selective scan backward",
          py::arg("u"), py::arg("delta"), py::arg("A"), py::arg("B"), py::arg("C"),
          py::arg("D"), py::arg("z"), py::arg("delta_bias"), py::arg("dout"),
          py::arg("x"), py::arg("out"), py::arg("dz"),
          py::arg("delta_softplus"), py::arg("recompute_out_z"),
          // None -> MAMBA_DETERMINISTIC env var, else torch.are_deterministic_algorithms_enabled()
          py::arg("deterministic") = py::none(),
          // Memory budget (MiB) for the per-channel dB/dC workspace in deterministic mode.
          py::arg("workspace_mb") = 256);''')
open(p, "w").write(s)

p = "setup.py"
s = open(p).read()
s = sub(s, '''                "csrc/selective_scan/selective_scan.cpp",
''', '''                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_det_reduce.cu",
''')
open(p, "w").write(s)
print("host edited")
