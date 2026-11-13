"""Compare kernel_outputs.py dumps.

    python compare.py stock1.pt stock2.pt patched_default.pt patched_det.pt

Columns are relative differences max|x - ref| / max|ref| against the first file (stock run 1):
stock run 2 (upstream's own run-to-run spread), patched default, patched deterministic.
"=" means bitwise identical.
"""
import sys
from collections import defaultdict

import torch

files = sys.argv[1:]
data = [torch.load(f) for f in files]
ref = data[0]
labels = ["stock#2", "patched-default", "patched-det"][: len(data) - 1]


def rel(a, b):
    if a is None and b is None:
        return "="
    if torch.equal(a, b):
        return "="
    d = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)
    return f"{d:.1e}"


rows = []
worst = defaultdict(float)
for key in ref:
    cells = [rel(d[key], ref[key]) for d in data[1:]]
    rows.append((key, cells))
w = max(len(k) for k in ref)
print(f"{'tensor':{w}s}  " + "  ".join(f"{l:>15s}" for l in labels))
for key, cells in rows:
    print(f"{key:{w}s}  " + "  ".join(f"{c:>15s}" for c in cells))

# Summary: does patched-default ever differ from stock where stock itself is bit-stable?
if len(data) >= 3:
    print("\nTensors where stock is bit-stable run-to-run but patched-default differs:")
    bad = [k for k, c in rows if c[0] == "=" and c[1] != "="]
    print("  none" if not bad else "\n".join(f"  {k}: {dict(rows)[k]}" for k in bad))
