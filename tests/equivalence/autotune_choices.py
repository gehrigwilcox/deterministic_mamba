"""Run the Mamba2 module fwd/bwd and record, alongside a digest of the outputs, which config every
autotuned Triton kernel picked. Used to attribute stock-vs-patched differences in *default* mode to
autotuning (timing-dependent config choice) rather than to the patches.

    python autotune_choices.py OUT.json [Mamba2|Mamba3]
"""
import hashlib
import json
import sys

import torch

import mamba_ssm.ops.triton as triton_ops
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3
import importlib
import pkgutil

DEV = "cuda"
torch.manual_seed(6)
layer = sys.argv[2] if len(sys.argv) > 2 else "Mamba2"
mod = {"Mamba2": Mamba2, "Mamba3": Mamba3}[layer](d_model=128, d_state=64, headdim=32, device=DEV)
x = torch.randn(4, 500, 128, generator=torch.Generator(device=DEV).manual_seed(6), device=DEV).requires_grad_()
y = mod(x)
y.backward(torch.randn(4, 500, 128, generator=torch.Generator(device=DEV).manual_seed(7), device=DEV))

h = hashlib.sha256()
for t in [y, x.grad] + [p.grad for p in mod.parameters()]:
    h.update(t.detach().float().cpu().numpy().tobytes())

choices = {}
for info in pkgutil.walk_packages(triton_ops.__path__, prefix="mamba_ssm.ops.triton."):
    try:
        m = importlib.import_module(info.name)
    except Exception:
        continue
    for name, obj in vars(m).items():
        if hasattr(obj, "best_config") and hasattr(obj, "configs"):
            try:
                cfg = obj.best_config
            except AttributeError:
                continue  # never launched
            if cfg is not None:
                choices[f"{info.name}.{name}"] = str(cfg)
json.dump({"digest": h.hexdigest(), "choices": choices}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(h.hexdigest()[:12], len(choices), "kernels")
