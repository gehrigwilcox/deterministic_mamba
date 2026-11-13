#include <torch/extension.h>
#include <vector>

#include "det_reduce.h"

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                   const c10::optional<at::Tensor> &D_,
                   const c10::optional<at::Tensor> &z_,
                   const c10::optional<at::Tensor> &delta_bias_,
                   bool delta_softplus);

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
                   int64_t workspace_mb);

at::Tensor
causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  bool silu_activation);

std::vector<at::Tensor>
causal_conv1d_bwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  at::Tensor &dout,
                  c10::optional<at::Tensor> &dx_,
                  bool silu_activation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("selective_scan_bwd", &selective_scan_bwd, "Deterministic selective scan backward");
    m.def("causal_conv1d_fwd", &causal_conv1d_fwd, "Causal conv1d forward (channel-first)");
    m.def("causal_conv1d_bwd", &causal_conv1d_bwd, "Deterministic causal conv1d backward (channel-first)");
    m.def("det_sum_dim0", &det_sum_dim0, "Fixed-order float32 sum over dim 0");
}
