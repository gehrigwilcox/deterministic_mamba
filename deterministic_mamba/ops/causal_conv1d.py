# Copyright (c) 2024, Tri Dao.
#
# Deterministic fork of causal_conv1d_interface.py (causal-conv1d 1.5.0.post8, as used by
# mamba_ssm 2.2.4). Channel-first layout only; seq_idx / initial / final states (Mamba2 features)
# are not supported. The backward kernel writes per-batch partial dweight/dbias and reduces them
# in a fixed order, so gradients are bitwise reproducible.

import torch
import torch.nn.functional as F

from deterministic_mamba import _C


class CausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation=None):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        if x.stride(2) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        ctx.activation = activation in ["silu", "swish"]
        out = _C.causal_conv1d_fwd(x, weight, bias, ctx.activation)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias = ctx.saved_tensors
        if dout.stride(2) != 1:
            dout = dout.contiguous()
        dx, dweight, dbias = _C.causal_conv1d_bwd(x, weight, bias, dout, None, ctx.activation)
        return dx, dweight, dbias if bias is not None else None, None


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if seq_idx is not None or initial_states is not None or return_final_states or final_states_out is not None:
        raise NotImplementedError(
            "deterministic_mamba.causal_conv1d_fn does not support seq_idx / initial_states / final_states"
        )
    return CausalConv1dFn.apply(x, weight, bias, activation)


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """Pure PyTorch reference (fp32 compute)."""
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
    out = out if activation is None else F.silu(out)
    return out.to(dtype=dtype_in)
