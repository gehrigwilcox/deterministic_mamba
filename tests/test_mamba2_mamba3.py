"""Mamba2 (upstream deterministic Triton kernels + patch 0002) and Mamba3 SISO (patches 0002, 0003)."""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import deterministic, n_unique

DEV = "cuda"


# --------------------------------------------------------------------------- Mamba2 SSD
def _ssd_inputs(batch=2, seqlen=1000, nheads=8, headdim=64, dstate=64, ngroups=2, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def r(*s, dt=dtype):
        return torch.randn(*s, generator=g, device=DEV).to(dt).requires_grad_()

    return dict(
        x=r(batch, seqlen, nheads, headdim),
        dt=(torch.rand(batch, seqlen, nheads, generator=g, device=DEV) * 0.1).to(dtype).requires_grad_(),
        A=(-torch.rand(nheads, generator=g, device=DEV) - 0.5).requires_grad_(),
        B=r(batch, seqlen, ngroups, dstate), C=r(batch, seqlen, ngroups, dstate),
        D=r(nheads, dt=torch.float32), z=r(batch, seqlen, nheads, headdim),
        dt_bias=r(nheads, dt=torch.float32),
    ), torch.randn(batch, seqlen, nheads, headdim, generator=g, device=DEV).to(dtype)


def _ssd_run(inp, dout, chunk_size=64):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    for v in inp.values():
        v.grad = None
    out = mamba_chunk_scan_combined(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], chunk_size,
                                    D=inp["D"], z=inp["z"], dt_bias=inp["dt_bias"], dt_softplus=True)
    out.backward(dout)
    return [out.detach().clone()] + [v.grad.clone() for v in inp.values()]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mamba2_ssd_bitwise_deterministic_and_close_to_default(dtype):
    inp, dout = _ssd_inputs(dtype=dtype)
    with deterministic(True):
        det = _ssd_run(inp, dout)
        assert n_unique(lambda: _ssd_run(inp, dout), runs=4) == 1
    with deterministic(False):
        default = _ssd_run(inp, dout)
    tol = 2e-3 if dtype == torch.float32 else 3e-2
    for a, b in zip(det, default):
        torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol * max(1.0, b.abs().max().item()))


def test_mamba2_ssd_matches_reference():
    from mamba_ssm.ops.triton.ssd_combined import ssd_selective_scan  # pure-PyTorch-ish reference path

    inp, dout = _ssd_inputs(seqlen=256, dtype=torch.float32)
    with deterministic(True):
        res = _ssd_run(inp, dout)
    ref_in = {k: v.detach().clone().requires_grad_() for k, v in inp.items()}
    out = ssd_selective_scan(ref_in["x"], ref_in["dt"], ref_in["A"], ref_in["B"], ref_in["C"], D=ref_in["D"],
                             z=ref_in["z"], dt_bias=ref_in["dt_bias"], dt_softplus=True)
    out.backward(dout)
    ref = [out.detach()] + [v.grad for v in ref_in.values()]
    for a, b in zip(res, ref):
        torch.testing.assert_close(a, b, rtol=5e-3, atol=5e-3 * max(1.0, b.abs().max().item()))


def test_toggle_after_import_selects_deterministic_configs():
    """Patch 0002: the config set follows the toggle at launch time, not at import time."""
    from mamba_ssm.ops.triton import ssd_chunk_scan
    from mamba_ssm.utils.determinism import deterministic_configs

    k = ssd_chunk_scan._chunk_scan_bwd_dc_kernel
    inp, dout = _ssd_inputs(seqlen=300)
    with deterministic(False):
        _ssd_run(inp, dout)
    with deterministic(True):
        _ssd_run(inp, dout)
        det_cfg = deterministic_configs(k.configs)
        assert len(det_cfg) == 1 and k.best_config.kwargs == det_cfg[0].kwargs
    assert len(k.configs) > 1  # full list still used for default mode


# --------------------------------------------------------------------------- Mamba3 SISO
def _m3_inputs(batch=3, seqlen=300, nheads=8, nheads_qk=2, headdim_qk=64, headdim_v=32, dtype=torch.float32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def r(*s, dt=dtype):
        return torch.randn(*s, generator=g, device=DEV).to(dt)

    a = -torch.empty(batch, nheads, seqlen, device=DEV).uniform_(1.0, 16.0, generator=g)
    dt = torch.exp(torch.rand(batch, nheads, seqlen, generator=g, device=DEV) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    inp = dict(
        Q=F.rms_norm(r(batch, seqlen, nheads_qk, headdim_qk), (headdim_qk,)),
        K=F.rms_norm(r(batch, seqlen, nheads_qk, headdim_qk), (headdim_qk,)),
        V=r(batch, seqlen, nheads, headdim_v), ADT=(a * dt).contiguous(), DT=dt.contiguous(),
        Trap=torch.rand(batch, nheads, seqlen, generator=g, device=DEV).to(dtype),
        Q_bias=r(nheads, headdim_qk), K_bias=r(nheads, headdim_qk),
        Angles=r(batch, seqlen, nheads, headdim_qk // 4, dt=torch.float32),
        D=torch.ones(nheads, device=DEV), Z=r(batch, seqlen, nheads, headdim_v),
    )
    for v in inp.values():
        v.requires_grad_()
    return inp


def _m3_run(inp, with_state_grad):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    for v in inp.values():
        v.grad = None
    out, ang, ssm, kst, vst = mamba3_siso_combined(**inp, chunk_size=64, return_final_states=True)
    loss = (out.float() ** 2).mean()
    if with_state_grad:  # exercises the dK state post-step (patch 0003)
        loss = loss + (kst.float() ** 2).sum() + (ssm.float() ** 2).mean() + (ang.float() ** 2).mean()
    loss.backward()
    return [out.detach().clone()] + [v.grad.clone() for v in inp.values()]


@pytest.mark.parametrize("with_state_grad", [False, True])
@pytest.mark.parametrize("nheads_qk", [8, 2])  # 2 -> GQA ratio 4
def test_mamba3_siso_bitwise_deterministic_and_close_to_default(with_state_grad, nheads_qk):
    inp = _m3_inputs(nheads_qk=nheads_qk)
    with deterministic(True):
        det = _m3_run(inp, with_state_grad)
        assert n_unique(lambda: _m3_run(inp, with_state_grad), runs=4) == 1
    with deterministic(False):
        default = _m3_run(inp, with_state_grad)
    # Both paths use the kernels' fast cos/sin/tanh approximations; with the large final-state loss
    # K.grad is only accurate to ~0.3% of its max in *either* mode (measured vs a float64 reference:
    # deterministic 0.077, default 0.098, max |grad| 29), so compare at 1% of max.
    for a, b in zip(det, default):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-3, atol=1e-2 * max(1.0, b.abs().max().item()))


def test_mamba3_dk_state_post_default_mode_is_atomic_and_nondeterministic():
    """The upstream dK/dK_bias atomics (GQA + batch reduction) in default mode."""
    inp = _m3_inputs(batch=16, nheads=32, nheads_qk=4)
    with deterministic(False):
        k = n_unique(lambda: [g for g in _m3_run(inp, True)[1:3]], runs=8)  # dK
    if k == 1:
        pytest.skip("atomics happened to be order-stable here")
    assert k > 1
