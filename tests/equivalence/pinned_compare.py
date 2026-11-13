"""Stock vs patched *default mode* with every autotuned Triton kernel pinned to the same config.

Autotuning picks configs by timing, independently in each build/process, and the config changes the
floating-point reduction order. To compare the patched code itself against stock, remove that
variable: pin every autotuned kernel to one config (the same deterministic rule in both builds), then
run module fwd/bwd and a short MixerModel training run, and save every tensor.

    python pinned_compare.py OUT.pt {Mamba1|Mamba2|Mamba3}
"""
import importlib
import pkgutil
import sys

import torch

import mamba_ssm.ops.triton as triton_ops
from mamba_ssm.utils.determinism import _estimate_config_cost  # present in stock 2.3.2.post1 too

out_path, layer = sys.argv[1], sys.argv[2]
torch.use_deterministic_algorithms(False)  # default mode

# ---- pin configs (stock: triton Autotuner; patched: mode-aware wrapper holding one in ._tuned)
pinned = 0
for info in pkgutil.walk_packages(triton_ops.__path__, prefix="mamba_ssm.ops.triton."):
    try:
        mod = importlib.import_module(info.name)
    except Exception:
        continue
    for obj in vars(mod).values():
        target = getattr(obj, "_tuned", obj)
        cfgs = getattr(target, "configs", None)
        if isinstance(cfgs, list) and cfgs and hasattr(target, "cache") and len(cfgs) > 1:
            target.configs = [min(cfgs, key=_estimate_config_cost)]
            pinned += 1

DEV = "cuda"
res = {}
from mamba_ssm.models.mixer_seq_simple import MixerModel  # noqa: E402
from mamba_ssm.modules.mamba2 import Mamba2  # noqa: E402
from mamba_ssm.modules.mamba3 import Mamba3  # noqa: E402
from mamba_ssm.modules.mamba_simple import Mamba  # noqa: E402

cls, kw = {"Mamba1": (Mamba, {}), "Mamba2": (Mamba2, dict(d_state=64, headdim=32)),
           "Mamba3": (Mamba3, dict(d_state=64, headdim=32))}[layer]
torch.manual_seed(6)
m = cls(d_model=128, **kw, device=DEV)
x = torch.randn(4, 500, 128, generator=torch.Generator(device=DEV).manual_seed(6), device=DEV).requires_grad_()
y = m(x)
y.backward(torch.randn(4, 500, 128, generator=torch.Generator(device=DEV).manual_seed(7), device=DEV))
res["module/out"] = y.detach().cpu()
res["module/dx"] = x.grad.cpu()
for k, p in m.named_parameters():
    res[f"module/d.{k}"] = p.grad.cpu()

# short training run (same task as train_curve.py)
if layer == "Mamba3":
    import mamba_ssm.models.mixer_seq_simple as mss
    mss.Mamba2 = Mamba3
cfg = {"Mamba1": {"layer": "Mamba1"}}.get(layer, {"layer": "Mamba2", "d_state": 64, "headdim": 64, "chunk_size": 64})
V, D, L, B = 64, 256, 256, 16
torch.manual_seed(0)
model = MixerModel(d_model=D, n_layer=4, d_intermediate=0, vocab_size=V, ssm_cfg=cfg, rms_norm=True,
                   fused_add_norm=True, residual_in_fp32=True, device=DEV)
head = torch.nn.Linear(D, V, bias=False, device=DEV)
opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-3)
g = torch.Generator().manual_seed(1234)
succ = torch.randint(0, V, (V, 4), generator=g)
losses = []
for step in range(60):
    seq = torch.empty(B, L // 2, dtype=torch.long)
    seq[:, 0] = torch.randint(0, V, (B,), generator=g)
    for t in range(1, L // 2):
        seq[:, t] = succ[seq[:, t - 1], torch.randint(0, 4, (B,), generator=g)]
    ids = torch.cat([seq, seq], 1).to(DEV)
    loss = torch.nn.functional.cross_entropy(head(model(ids[:, :-1])).float().reshape(-1, V), ids[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(loss.detach())
res["train/losses"] = torch.stack(losses).cpu()
for i, p in enumerate(model.parameters()):
    res[f"train/param{i}"] = p.detach().cpu()
torch.save(res, out_path)
print(f"{layer}: pinned {pinned} kernels, final loss {losses[-1].item():.6f}")
