"""Mamba1 (CUDA selective scan, patch 0001) + causal_conv1d (patch causal-conv1d/0001)."""

import pytest
import torch
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import (
    mamba_inner_fn,
    mamba_inner_ref,
    selective_scan_fn,
    selective_scan_ref,
)
from mamba_ssm.utils.determinism import set_deterministic_mode

from conftest import deterministic, n_unique

DEV = "cuda"
TOL = {torch.float32: (6e-4, 2e-3), torch.float16: (3e-3, 5e-3), torch.bfloat16: (3e-2, 5e-2)}


def make_inputs(batch, dim, seqlen, dstate, itype, vb, vc, ngroups, wtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    is_complex = wtype == torch.complex64

    def r(*shape, dtype=itype):
        return torch.randn(*shape, generator=g, device=DEV).to(dtype)

    if is_complex:
        A = torch.complex(-0.5 * torch.rand(dim, dstate, generator=g, device=DEV), r(dim, dstate, dtype=torch.float32))
    else:
        A = -0.5 * torch.rand(dim, dstate, generator=g, device=DEV)
    L = seqlen * (2 if is_complex else 1)

    def bc(var):
        if var:
            return r(batch, ngroups, dstate, L) if ngroups > 1 else r(batch, dstate, L)
        return (torch.complex(r(dim, dstate, dtype=torch.float32), r(dim, dstate, dtype=torch.float32))
                if is_complex else r(dim, dstate, dtype=torch.float32))

    inputs = dict(
        u=r(batch, dim, seqlen), delta=(0.5 * torch.rand(batch, dim, seqlen, generator=g, device=DEV)).to(itype),
        A=A, B=bc(vb), C=bc(vc), D=r(dim, dtype=torch.float32), z=r(batch, dim, seqlen),
        delta_bias=0.5 * torch.rand(dim, generator=g, device=DEV),
    )
    for v in inputs.values():
        v.requires_grad_()
    return inputs, r(batch, dim, seqlen)


def run(fn, inputs, dout):
    for v in inputs.values():
        v.grad = None
    out = fn(inputs["u"], inputs["delta"], inputs["A"], inputs["B"], inputs["C"], inputs["D"],
             z=inputs["z"], delta_bias=inputs["delta_bias"], delta_softplus=True)
    out.backward(dout)
    return [out.detach().clone()] + [inputs[k].grad.clone() for k in inputs]


CASES = [(True, True, 1), (True, True, 2), (False, True, 1), (True, False, 1), (False, False, 1)]


@pytest.mark.parametrize("det", [False, True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seqlen", [128, 1024, 4096 + 17])
@pytest.mark.parametrize("vb,vc,ngroups", CASES)
def test_selective_scan_matches_reference(det, itype, seqlen, vb, vc, ngroups):
    inputs, dout = make_inputs(2, 8, seqlen, 8, itype, vb, vc, ngroups)
    with deterministic(det):
        res = run(selective_scan_fn, inputs, dout)
    ref_in = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
    ref = run(selective_scan_ref, ref_in, dout)
    rtol, atol = TOL[itype]
    scale = 1.0 if itype == torch.float32 else 3.0
    for name, a, b in zip(["out"] + list(inputs), res, ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=rtol * scale,
                                   atol=atol * scale * max(1.0, b.abs().max().item() * 0.05),
                                   msg=lambda m: f"{name}: {m}")


@pytest.mark.parametrize("vb,vc", [(True, True), (False, False), (True, False)])
def test_complex_deterministic_matches_reference(vb, vc):
    inputs, dout = make_inputs(2, 8, 300, 4, torch.float32, vb, vc, 1, wtype=torch.complex64)
    with deterministic(True):
        res = run(selective_scan_fn, inputs, dout)
        again = run(selective_scan_fn, inputs, dout)
    ref = run(selective_scan_ref, {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}, dout)
    for a, b, c in zip(res, ref, again):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)
        assert torch.equal(a, c)


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vb,vc,ngroups", CASES)
def test_selective_scan_bitwise_deterministic_when_enabled(itype, vb, vc, ngroups):
    inputs, dout = make_inputs(4, 256, 1500, 16, itype, vb, vc, ngroups)
    with deterministic(True):
        assert n_unique(lambda: run(selective_scan_fn, inputs, dout)) == 1


def test_default_mode_is_upstream_atomic_path():
    """Off by default: identical inputs give varying dB (upstream behavior, kept for speed)."""
    inputs, dout = make_inputs(4, 512, 1500, 16, torch.float32, True, True, 1)
    with deterministic(False):
        k = n_unique(lambda: run(selective_scan_fn, inputs, dout)[4:5], runs=8)
    if k == 1:
        pytest.skip("atomics happened to be order-stable on this GPU/shape")
    assert k > 1


@pytest.mark.parametrize("how", ["torch", "override", "env"])
def test_toggle_mechanisms(how, monkeypatch):
    inputs, dout = make_inputs(4, 256, 1500, 16, torch.float32, True, True, 1)
    if how == "torch":
        ctx = deterministic(True)
    elif how == "override":
        set_deterministic_mode(True)
        ctx = deterministic(False)
    else:
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        ctx = deterministic(False)
    try:
        with ctx:
            assert n_unique(lambda: run(selective_scan_fn, inputs, dout)) == 1
    finally:
        set_deterministic_mode(None)


@pytest.mark.parametrize("vb,vc,ngroups", [(True, True, 1), (True, True, 4), (True, False, 2)])
def test_bits_independent_of_workspace_budget(vb, vc, ngroups, monkeypatch):
    inputs, dout = make_inputs(2, 64, 3000, 16, torch.float32, vb, vc, ngroups)
    with deterministic(True):
        monkeypatch.setenv("MAMBA_DETERMINISTIC_WORKSPACE_MB", "4096")
        big = run(selective_scan_fn, inputs, dout)
        monkeypatch.setenv("MAMBA_DETERMINISTIC_WORKSPACE_MB", "1")
        small = run(selective_scan_fn, inputs, dout)
    for a, b in zip(big, small):
        assert torch.equal(a, b)


def _inner_inputs(batch=2, seqlen=777, d_model=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    d_inner, d_state, dt_rank = 2 * d_model, 16, 4

    def r(*shape, s=1.0):
        return (torch.randn(*shape, generator=g, device=DEV) * s).requires_grad_()

    return [r(batch, 2 * d_inner, seqlen), r(d_inner, 1, 4, s=0.5), r(d_inner, s=0.1),
            r(dt_rank + 2 * d_state, d_inner, s=d_inner ** -0.5), r(d_inner, dt_rank, s=0.5),
            r(d_model, d_inner, s=d_inner ** -0.5), None,
            (-torch.rand(d_inner, d_state, generator=g, device=DEV)).requires_grad_(), None, None,
            r(d_inner), (torch.rand(d_inner, generator=g, device=DEV) - 4).requires_grad_()]


def _inner_run(fn, args, dout):
    for a in args:
        if a is not None:
            a.grad = None
    out = fn(*args[:11], delta_bias=args[11], delta_softplus=True)
    out.backward(dout)
    return [out.detach().clone()] + [a.grad.clone() for a in args if a is not None]


def test_mamba_inner_fn_reference_and_determinism():
    args = _inner_inputs()
    dout = torch.randn(2, 777, 64, device=DEV)
    with deterministic(True):
        res = _inner_run(mamba_inner_fn, args, dout)
        assert n_unique(lambda: _inner_run(mamba_inner_fn, args, dout), runs=3) == 1
    ref = _inner_run(mamba_inner_ref, [a.detach().clone().requires_grad_() if a is not None else None for a in args], dout)
    for a, b in zip(res, ref):
        torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------- causal_conv1d
from causal_conv1d import causal_conv1d_fn  # noqa: E402


@pytest.mark.parametrize("det", [False, True])
@pytest.mark.parametrize("seqlen", [129, 2049, 4097])
@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 4])
def test_causal_conv1d_silu_chunk_boundary_fix(det, seqlen, itype, width):
    """seqlen 2049 / 4097 hit the upstream boundary bug (wrong dx/dweight/dbias)."""
    torch.manual_seed(0)
    x = torch.randn(2, 128, seqlen, device=DEV, dtype=itype)[:, :64].requires_grad_()
    w = torch.randn(64, width, device=DEV).requires_grad_()
    b = torch.randn(64, device=DEV).requires_grad_()
    dout = torch.randn(2, 64, seqlen, device=DEV, dtype=itype)
    with deterministic(det):
        causal_conv1d_fn(x, w, b, activation="silu").backward(dout)
    xr, wr, br = (t.detach().float().requires_grad_() for t in (x, w, b))
    F.silu(F.conv1d(xr, wr.unsqueeze(1), br, padding=width - 1, groups=64)[..., :seqlen]).backward(dout.float())
    tol = dict(rtol=3e-4, atol=1e-3) if itype == torch.float32 else dict(rtol=3e-2, atol=1e-1)
    for g, gr in [(x.grad, xr.grad), (w.grad, wr.grad), (b.grad, br.grad)]:
        torch.testing.assert_close(g.float(), gr, **tol)
