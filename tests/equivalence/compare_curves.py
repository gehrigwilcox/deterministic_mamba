"""Compare train_curve.py dumps for one layer type.

    python compare_curves.py stock1.pt stock2.pt patched_default.pt patched_det.pt
"""
import sys

import torch

runs = [torch.load(f) for f in sys.argv[1:]]
labels = ["stock#2", "patched-default", "patched-det"][: len(runs) - 1]
ref = runs[0]
L0 = torch.tensor(ref["losses"])
print(f"  stock#1: loss {L0[0]:.3f} -> {L0[-1]:.4f} (mean of last 20 steps {L0[-20:].mean():.4f})")
for lab, r in zip(labels, runs[1:]):
    L = torch.tensor(r["losses"])
    same = torch.equal(L, L0)
    first_div = int((L != L0).nonzero()[0]) if not same else None
    pdiff = max(((a - b).abs().max() / b.abs().max()).item() for a, b in zip(r["params"], ref["params"]))
    print(f"  {lab:16s} final {L[-1]:.4f} (last-20 mean {L[-20:].mean():.4f})  "
          f"max|dloss| {(L - L0).abs().max():.2e}  "
          f"{'identical curve' if same else f'first differs at step {first_div}'}  "
          f"max rel weight diff {pdiff:.1e}")
