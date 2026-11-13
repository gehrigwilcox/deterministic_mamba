"""Build-time helper: restrict an upstream setup.py's gencode list to CUDA_ARCHS (e.g. "75" or
"80;90") to cut compile time. Applied to build trees only; not part of the patches."""
import os, sys

path = sys.argv[1]
archs = [a.strip().replace(".", "") for a in os.environ["CUDA_ARCHS"].replace(",", ";").split(";") if a.strip()]
s = open(path).read()
anchor = "\n    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
assert s.count(anchor) == 1, f"{path}: layout changed, cannot limit archs"
gencode = []
for a in archs:
    gencode += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
inject = (
    "\n    # [deterministic-mamba build] restrict target architectures (CUDA_ARCHS)\n"
    f"    cc_flag = [f for f in cc_flag if f != '-gencode' and not f.startswith('arch=')] + {gencode!r}\n"
)
s = s.replace(anchor, inject + anchor, 1)
open(path, "w").write(s)
print(f"{path}: gencode -> {archs}")
