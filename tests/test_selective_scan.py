import pytest
import torch
from einops import rearrange

import deterministic_mamba
from deterministic_mamba.ops.selective_scan_interface import (
    mamba_inner_fn,
    mamba_inner_ref,
    selective_scan_fn,
    selective_scan_ref,
)

DEVICE = "cuda"
TOL = {torch.float32: (6e-4, 2e-3), torch.float16: (3e-3, 5e-3), torch.bfloat16: (3e-2, 5e-2)}


def _make_inputs(batch, dim, seqlen, dstate, itype, is_variable_B, is_variable_C, ngroups, has_D,
                 has_z, has_delta_bias, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def r(*shape, dtype=itype, scale=1.0):
        return (torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.float32) * scale).to(dtype)

    A = (-0.5 * torch.rand(dim, dstate, generator=g, device=DEVICE)).requires_grad_()
    if is_variable_B:
        B = r(batch, ngroups, dstate, seqlen) if ngroups > 1 else r(batch, dstate, seqlen)
    else:
        B = r(dim, dstate, dtype=torch.float32)
    if is_variable_C:
        C = r(batch, ngroups, dstate, seqlen) if ngroups > 1 else r(batch, dstate, seqlen)
    else:
        C = r(dim, dstate, dtype=torch.float32)
    B.requires_grad_()
    C.requires_grad_()
    D = r(dim, dtype=torch.float32).requires_grad_() if has_D else None
    z = r(batch, dim, seqlen).requires_grad_() if has_z else None
    delta_bias = (0.5 * torch.rand(dim, generator=g, device=DEVICE)).requires_grad_() if has_delta_bias else None
    u = r(batch, dim, seqlen).requires_grad_()
    delta = (0.5 * torch.rand(batch, dim, seqlen, generator=g, device=DEVICE)).to(itype).requires_grad_()
    dout = r(batch, dim, seqlen)
    return dict(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z, delta_bias=delta_bias), dout


def _grads(fn, inputs, dout, **kw):
    for v in inputs.values():
        if v is not None:
            v.grad = None
    out = fn(inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"], inputs["D"],
             z=inputs["z"], delta_bias=inputs["delta_bias"], delta_softplus=True, **kw)
    out.backward(dout)
    return out.detach().clone(), {k: v.grad.clone() for k, v in inputs.items() if v is not None}


CASES = [
    # (is_variable_B, is_variable_C, ngroups)
    (True, True, 1),
    (True, True, 2),
    (False, True, 1),
    (True, False, 1),
    (False, False, 1),
]


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [64, 128, 511, 1024, 2048, 4096 + 17])
@pytest.mark.parametrize("vb,vc,ngroups", CASES)
@pytest.mark.parametrize("has_z", [False, True])
def test_selective_scan_matches_reference(itype, seqlen, vb, vc, ngroups, has_z):
    torch.manual_seed(0)
    batch, dim, dstate = 2, 8, 8
    inputs, dout = _make_inputs(batch, dim, seqlen, dstate, itype, vb, vc, ngroups, True, has_z, True)
    out, grads = _grads(selective_scan_fn, inputs, dout)

    ref_inputs = {k: (v.detach().clone().requires_grad_() if v is not None else None) for k, v in inputs.items()}
    out_ref, grads_ref = _grads(selective_scan_ref, ref_inputs, dout)

    rtol, atol = TOL[itype]
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=rtol, atol=atol)
    for k in grads:
        # dB/dC/dA sum over many terms; give them a bit more slack in low precision.
        scale = 1.0 if itype == torch.float32 else 3.0
        torch.testing.assert_close(grads[k].float(), grads_ref[k].float(), rtol=rtol * scale,
                                   atol=atol * scale * max(1.0, grads_ref[k].abs().max().item() * 0.05),
                                   msg=lambda m: f"grad {k}: {m}")


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vb,vc,ngroups", CASES)
def test_selective_scan_backward_is_bitwise_deterministic(itype, vb, vc, ngroups):
    # Large enough that the upstream atomicAdd version is visibly non-deterministic.
    batch, dim, seqlen, dstate = 4, 256, 1500, 16
    inputs, dout = _make_inputs(batch, dim, seqlen, dstate, itype, vb, vc, ngroups, True, True, True)
    _, first = _grads(selective_scan_fn, inputs, dout)
    for _ in range(5):
        _, again = _grads(selective_scan_fn, inputs, dout)
        for k in first:
            assert torch.equal(first[k], again[k]), f"grad {k} differs between identical runs"


@pytest.mark.parametrize("vb,vc,ngroups", [(True, True, 1), (True, True, 4), (True, False, 2)])
def test_result_independent_of_workspace_budget(vb, vc, ngroups):
    """Slicing channels to bound memory must not change a single bit."""
    batch, dim, seqlen, dstate = 2, 64, 3000, 16
    inputs, dout = _make_inputs(batch, dim, seqlen, dstate, torch.float32, vb, vc, ngroups, True, True, True)
    old = deterministic_mamba.workspace_mb()
    try:
        deterministic_mamba.set_workspace_mb(4096)  # one slice per group
        _, big = _grads(selective_scan_fn, inputs, dout)
        deterministic_mamba.set_workspace_mb(1)  # 1 MiB -> many slices
        _, small = _grads(selective_scan_fn, inputs, dout)
    finally:
        deterministic_mamba.set_workspace_mb(old)
    for k in big:
        assert torch.equal(big[k], small[k]), f"grad {k} depends on the workspace budget"


def _inner_inputs(batch, seqlen, d_model, itype, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    d_inner, d_state, dt_rank, d_conv = 2 * d_model, 16, max(1, d_model // 16), 4

    def r(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g, device=DEVICE) * scale).to(dtype).requires_grad_()

    return dict(
        xz=r(batch, 2 * d_inner, seqlen, dtype=itype),
        conv1d_weight=r(d_inner, 1, d_conv, scale=0.5),
        conv1d_bias=r(d_inner, scale=0.1),
        x_proj_weight=r(dt_rank + 2 * d_state, d_inner, scale=d_inner ** -0.5),
        delta_proj_weight=r(d_inner, dt_rank, scale=dt_rank ** -0.5),
        out_proj_weight=r(d_model, d_inner, scale=d_inner ** -0.5),
        out_proj_bias=None,
        A=(-torch.rand(d_inner, d_state, generator=g, device=DEVICE)).requires_grad_(),
        D=r(d_inner),
        delta_bias=(torch.rand(d_inner, generator=g, device=DEVICE) - 4).requires_grad_(),
    )


def _inner_run(fn, inp, dout):
    for v in inp.values():
        if v is not None:
            v.grad = None
    out = fn(inp["xz"], inp["conv1d_weight"], inp["conv1d_bias"], inp["x_proj_weight"],
             inp["delta_proj_weight"], inp["out_proj_weight"], inp["out_proj_bias"], inp["A"],
             None, None, inp["D"], delta_bias=inp["delta_bias"], delta_softplus=True)
    out.backward(dout)
    return out.detach().clone(), {k: v.grad.clone() for k, v in inp.items() if v is not None}


def test_mamba_inner_fn_matches_reference_and_is_deterministic():
    batch, seqlen, d_model = 2, 777, 64
    inp = _inner_inputs(batch, seqlen, d_model, torch.float32)
    dout = torch.randn(batch, seqlen, d_model, device=DEVICE)
    out, grads = _inner_run(mamba_inner_fn, inp, dout)
    ref_inp = {k: (v.detach().clone().requires_grad_() if v is not None else None) for k, v in inp.items()}
    out_ref, grads_ref = _inner_run(mamba_inner_ref, ref_inp, dout)
    torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)
    for k in grads:
        torch.testing.assert_close(grads[k], grads_ref[k], rtol=2e-3, atol=2e-3, msg=lambda m: f"{k}: {m}")
    for _ in range(3):
        _, again = _inner_run(mamba_inner_fn, inp, dout)
        for k in grads:
            assert torch.equal(grads[k], again[k]), k
