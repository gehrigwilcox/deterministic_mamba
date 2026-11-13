import pytest
import torch

from deterministic_mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_ref

DEVICE = "cuda"


def _run(fn, x, w, b, act, dout):
    for t in (x, w, b):
        if t is not None:
            t.grad = None
    out = fn(x, w, b, activation=act)
    out.backward(dout)
    return out.detach().clone(), [t.grad.clone() for t in (x, w, b) if t is not None]


@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("seqlen", [1, 8, 129, 2049])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("act", [None, "silu"])
def test_causal_conv1d_matches_reference(itype, wtype, width, seqlen, has_bias, act):
    torch.manual_seed(0)
    batch, dim = 2, 64
    # Non-contiguous channel slice, like mamba_inner_fn's xz.chunk(2, dim=1)
    x = torch.randn(batch, 2 * dim, seqlen, device=DEVICE, dtype=itype)[:, :dim].requires_grad_()
    w = torch.randn(dim, width, device=DEVICE, dtype=wtype).requires_grad_()
    b = torch.randn(dim, device=DEVICE, dtype=wtype).requires_grad_() if has_bias else None
    dout = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=itype)
    out, grads = _run(causal_conv1d_fn, x, w, b, act, dout)

    xr = x.detach().float().requires_grad_()
    wr = w.detach().float().requires_grad_()
    br = b.detach().float().requires_grad_() if b is not None else None
    out_ref, grads_ref = _run(causal_conv1d_ref, xr, wr, br, act, dout.float())
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 and wtype == torch.float32 else (3e-2, 5e-2)
    torch.testing.assert_close(out.float(), out_ref, rtol=rtol, atol=atol)
    for g, gr in zip(grads, grads_ref):
        torch.testing.assert_close(g.float(), gr, rtol=rtol, atol=atol * max(1.0, gr.abs().max().item() * 0.02))


@pytest.mark.parametrize("itype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_backward_is_bitwise_deterministic(itype):
    batch, dim, seqlen = 64, 512, 1024
    x = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=itype).requires_grad_()
    w = torch.randn(dim, 4, device=DEVICE).requires_grad_()
    b = torch.randn(dim, device=DEVICE).requires_grad_()
    dout = torch.randn(batch, dim, seqlen, device=DEVICE, dtype=itype)
    _, first = _run(causal_conv1d_fn, x, w, b, "silu", dout)
    for _ in range(5):
        _, again = _run(causal_conv1d_fn, x, w, b, "silu", dout)
        for g0, g1 in zip(first, again):
            assert torch.equal(g0, g1)
