"""Train a MixerModel on a learnable copy task and dump the loss curve + final weights.

    python train_curve.py OUT.pt {Mamba1|Mamba2|Mamba3} {default|det} [steps]

Every run uses identical init and data; only the build / mode differs.
"""
import sys

import torch

from mamba_ssm.models.mixer_seq_simple import MixerModel

out_path, layer, mode = sys.argv[1:4]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
torch.use_deterministic_algorithms(mode == "det")  # after import, like a user would

CFG = {"Mamba1": {"layer": "Mamba1"},
       "Mamba2": {"layer": "Mamba2", "d_state": 64, "headdim": 64, "chunk_size": 64},
       # Stock MixerModel can't build Mamba3 layers (that's patch 0004), so build Mamba3 models the
       # same way in every env: put Mamba3 in the Mamba2 slot of create_block.
       "Mamba3": {"layer": "Mamba2", "d_state": 64, "headdim": 64, "chunk_size": 64}}
if layer == "Mamba3":
    import mamba_ssm.models.mixer_seq_simple as mss
    from mamba_ssm.modules.mamba3 import Mamba3

    mss.Mamba2 = Mamba3
V, D, L, B = 64, 256, 256, 16
torch.manual_seed(0)
model = MixerModel(d_model=D, n_layer=4, d_intermediate=0, vocab_size=V, ssm_cfg=dict(CFG[layer]),
                   rms_norm=True, fused_add_norm=True, residual_in_fp32=True, device="cuda")
head = torch.nn.Linear(D, V, bias=False, device="cuda")
params = list(model.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=1e-3)
g = torch.Generator().manual_seed(1234)
# Learnable synthetic language: a fixed sparse bigram table (each token has 4 possible successors,
# so the achievable loss on that part is ~ln 4), followed by a copy of the first half (long-range).
succ = torch.randint(0, V, (V, 4), generator=g)


def sample_batch():
    seq = torch.empty(B, L // 2, dtype=torch.long)
    seq[:, 0] = torch.randint(0, V, (B,), generator=g)
    for t in range(1, L // 2):
        seq[:, t] = succ[seq[:, t - 1], torch.randint(0, 4, (B,), generator=g)]
    return torch.cat([seq, seq], 1).cuda()


losses = []
for step in range(steps):
    ids = sample_batch()
    logits = head(model(ids[:, :-1])).float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(loss.item())
torch.save({"losses": losses, "params": [p.detach().cpu() for p in params]}, out_path)
print(f"{layer} {mode}: loss {losses[0]:.3f} -> {losses[-1]:.4f}")
