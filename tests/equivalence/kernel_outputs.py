"""Dump outputs + gradients of every Mamba entry point for one process, for cross-build comparison.

    python kernel_outputs.py OUT.pt {default|det}

Run it once per build/mode (stock upstream, patched default, patched deterministic) and compare the
files with compare.py. Inputs are seeded, so all runs see identical inputs. The mode is switched
*after* importing mamba_ssm, the way a user would.
"""
import math
import sys

import torch
import torch.nn.functional as F

import mamba_ssm  # noqa: F401  (import before toggling, deliberately)
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

out_path, mode = sys.argv[1], sys.argv[2]
torch.use_deterministic_algorithms(mode == "det")
DEV = "cuda"
results = {}


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def record(prefix, named):
    for k, v in named.items():
        results[f"{prefix}/{k}"] = v.detach().float().cpu().clone() if v is not None else None


def grads(named):
    return {f"d{k}": v.grad for k, v in named.items() if isinstance(v, torch.Tensor) and v.grad is not None}


# ------------------------------------------------------------------ Mamba1 selective scan
for dtype in (torch.float32, torch.bfloat16):
    g = gen(1)
    b, d, L, n = 4, 256, 1500, 16
    inp = dict(u=torch.randn(b, d, L, generator=g, device=DEV).to(dtype),
               delta=(0.5 * torch.rand(b, d, L, generator=g, device=DEV)).to(dtype),
               A=-0.5 * torch.rand(d, n, generator=g, device=DEV),
               B=torch.randn(b, n, L, generator=g, device=DEV).to(dtype),
               C=torch.randn(b, n, L, generator=g, device=DEV).to(dtype),
               D=torch.randn(d, generator=g, device=DEV), z=torch.randn(b, d, L, generator=g, device=DEV).to(dtype),
               delta_bias=0.5 * torch.rand(d, generator=g, device=DEV))
    for v in inp.values():
        v.requires_grad_()
    y = selective_scan_fn(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], z=inp["z"],
                          delta_bias=inp["delta_bias"], delta_softplus=True)
    y.backward(torch.randn(b, d, L, generator=g, device=DEV).to(dtype))
    record(f"m1_scan_{str(dtype)[6:]}", {"out": y, **grads(inp)})

# ------------------------------------------------------------------ Mamba1 mamba_inner_fn (conv + scan + projections)
g = gen(2)
dm, di, ns, r, L, b = 64, 128, 16, 4, 777, 2
args = [torch.randn(b, 2 * di, L, generator=g, device=DEV), 0.5 * torch.randn(di, 1, 4, generator=g, device=DEV),
        0.1 * torch.randn(di, generator=g, device=DEV), torch.randn(r + 2 * ns, di, generator=g, device=DEV) * di ** -0.5,
        0.5 * torch.randn(di, r, generator=g, device=DEV), torch.randn(dm, di, generator=g, device=DEV) * di ** -0.5,
        None, -torch.rand(di, ns, generator=g, device=DEV), None, None, torch.randn(di, generator=g, device=DEV),
        torch.rand(di, generator=g, device=DEV) - 4]
for a in args:
    if a is not None:
        a.requires_grad_()
y = mamba_inner_fn(*args[:11], delta_bias=args[11], delta_softplus=True)
y.backward(torch.randn(b, L, dm, generator=g, device=DEV))
record("m1_inner", {"out": y, **{f"darg{i}": a.grad for i, a in enumerate(args) if a is not None}})

# ------------------------------------------------------------------ causal_conv1d (channel-first, as Mamba1 uses it)
for L in (1000, 2049):
    g = gen(3)
    x = torch.randn(4, 128, L, generator=g, device=DEV).requires_grad_()
    w = torch.randn(128, 4, generator=g, device=DEV).requires_grad_()
    bias = torch.randn(128, generator=g, device=DEV).requires_grad_()
    y = causal_conv1d_fn(x, w, bias, activation="silu")
    y.backward(torch.randn(4, 128, L, generator=g, device=DEV))
    record(f"conv_L{L}", {"out": y, "dx": x.grad, "dw": w.grad, "db": bias.grad})

# ------------------------------------------------------------------ Mamba2 SSD
for dtype in (torch.float32, torch.bfloat16):
    g = gen(4)
    b, L, h, p, n, ng = 2, 1000, 8, 64, 64, 2
    inp = dict(x=torch.randn(b, L, h, p, generator=g, device=DEV).to(dtype),
               dt=(0.1 * torch.rand(b, L, h, generator=g, device=DEV)).to(dtype),
               A=-torch.rand(h, generator=g, device=DEV) - 0.5,
               B=torch.randn(b, L, ng, n, generator=g, device=DEV).to(dtype),
               C=torch.randn(b, L, ng, n, generator=g, device=DEV).to(dtype),
               D=torch.randn(h, generator=g, device=DEV), z=torch.randn(b, L, h, p, generator=g, device=DEV).to(dtype),
               dt_bias=torch.randn(h, generator=g, device=DEV))
    for v in inp.values():
        v.requires_grad_()
    y = mamba_chunk_scan_combined(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], 64, D=inp["D"], z=inp["z"],
                                  dt_bias=inp["dt_bias"], dt_softplus=True)
    y.backward(torch.randn(b, L, h, p, generator=g, device=DEV).to(dtype))
    record(f"m2_ssd_{str(dtype)[6:]}", {"out": y, **grads(inp)})

# ------------------------------------------------------------------ Mamba3 SISO (incl. final-state gradients)
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined  # noqa: E402

g = gen(5)
b, L, h, hqk, dqk, dv = 3, 300, 8, 2, 64, 32
a = -torch.empty(b, h, L, device=DEV).uniform_(1.0, 16.0, generator=g)
dt = torch.exp(torch.rand(b, h, L, generator=g, device=DEV) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
inp = dict(Q=F.rms_norm(torch.randn(b, L, hqk, dqk, generator=g, device=DEV), (dqk,)),
           K=F.rms_norm(torch.randn(b, L, hqk, dqk, generator=g, device=DEV), (dqk,)),
           V=torch.randn(b, L, h, dv, generator=g, device=DEV), ADT=(a * dt).contiguous(), DT=dt.contiguous(),
           Trap=torch.rand(b, h, L, generator=g, device=DEV), Q_bias=torch.randn(h, dqk, generator=g, device=DEV),
           K_bias=torch.randn(h, dqk, generator=g, device=DEV),
           Angles=torch.randn(b, L, h, dqk // 4, generator=g, device=DEV),
           D=torch.ones(h, device=DEV), Z=torch.randn(b, L, h, dv, generator=g, device=DEV))
for v in inp.values():
    v.requires_grad_()
y, ang, ssm, kst, vst = mamba3_siso_combined(**inp, chunk_size=64, return_final_states=True)
((y ** 2).mean() + (kst ** 2).sum() + (ssm ** 2).mean() + (ang ** 2).mean()).backward()
record("m3_siso", {"out": y, "kstate": kst, **grads(inp)})

# ------------------------------------------------------------------ full modules (what MixerModel calls)
from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402

for name, cls, kw in [("m1_module", Mamba, {}), ("m2_module", Mamba2, dict(d_state=64, headdim=32)),
                      ("m3_module", Mamba3, dict(d_state=64, headdim=32))]:
    torch.manual_seed(6)
    mod = cls(d_model=128, **kw, device=DEV)
    x = torch.randn(4, 500, 128, generator=gen(6), device=DEV).requires_grad_()
    y = mod(x)
    y.backward(torch.randn(4, 500, 128, generator=gen(7), device=DEV))
    record(name, {"out": y, "dx": x.grad, **{f"d.{k}": p.grad for k, p in mod.named_parameters()}})

torch.save(results, out_path)
print(f"saved {len(results)} tensors ({mode}) -> {out_path}")
