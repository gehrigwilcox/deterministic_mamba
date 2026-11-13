#!/usr/bin/env bash
# Fetch the pinned upstream tags, apply patches/, and pip-install the patched packages
# (they keep their upstream names: `causal_conv1d` and `mamba_ssm`).
#
#   scripts/build.sh                 # fetch + patch + install
#   scripts/build.sh --patch-only    # fetch + patch into $BUILD_DIR, don't install
#   scripts/build.sh --from-work     # dev: build from work/<repo> instead of fetching
#   scripts/build.sh --check         # only run the environment checks
#
# Env:
#   CUDA_ARCHS="80;90"     only compile these SM architectures (default: upstream's full list)
#   BUILD_DIR=...          where sources are staged (default: ./build)
#   PIP="python -m pip"    pip of your training env
#   PYTHON=python          python of your training env (default: taken from PIP if it is "<py> -m pip")
#   ALLOW_CUDA_MISMATCH=1  build even if nvcc's major.minor differs from torch's CUDA (not advised)
#
# The packages are installed with --no-deps so pip never replaces your torch/triton. Their runtime
# dependencies are checked (not installed) below; install any that are reported missing yourself.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/versions.env"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PIP="${PIP:-python -m pip}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ "$PIP" == *" -m pip" ]]; then PYTHON="${PIP% -m pip}"; else PYTHON=python; fi
fi
MODE="${1:-install}"

preflight() {
    echo ">> preflight checks ($PYTHON)"
    "$PYTHON" - <<'EOF'
import importlib.metadata as md, os, re, shutil, subprocess, sys
from packaging.version import Version

errors, warnings = [], []
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch is not installed in this environment")
tv = Version(torch.__version__.split("+")[0])
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda})")
if tv < Version("2.8.0"):
    errors.append(f"torch {tv} < 2.8.0: `import mamba_ssm` fails on older torch (its dependency "
                  "quack-kernels uses torch.float4_e2m1fn_x2), even if you only use Mamba1.")
try:
    import triton
    trv = Version(triton.__version__)
    print(f"   triton {triton.__version__}")
    if trv < Version("3.5.0"):
        warnings.append(f"triton {trv} < 3.5.0 (mamba_ssm requires >= 3.5.0): the Mamba3 Triton "
                        "kernels will not compile. Mamba1/Mamba2 work. torch 2.8 pins triton 3.4; "
                        "use torch >= 2.9 for Mamba3.")
except ImportError:
    errors.append("triton is not installed")

nvcc = shutil.which("nvcc", path=os.path.join(os.environ.get("CUDA_HOME", ""), "bin") + os.pathsep + os.environ.get("PATH", ""))
if nvcc is None:
    errors.append("nvcc not found (set CUDA_HOME or PATH)")
elif torch.version.cuda:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+)\.(\d+)", out)
    nv = f"{m.group(1)}.{m.group(2)}" if m else "?"
    tc = ".".join(torch.version.cuda.split(".")[:2])
    print(f"   nvcc {nv} ({nvcc})")
    if nv != tc:
        msg = (f"nvcc {nv} != torch's CUDA {tc}. Match major AND minor: newer toolkits' libcu++ can "
               "fail to compile against older torch builds even within the same major version.")
        (warnings if os.environ.get("ALLOW_CUDA_MISMATCH") == "1" else errors).append(
            msg + ("" if os.environ.get("ALLOW_CUDA_MISMATCH") == "1" else " Set ALLOW_CUDA_MISMATCH=1 to try anyway."))

# Runtime deps of mamba_ssm / causal_conv1d (reported, never installed: we install with --no-deps).
reqs = ["packaging", "ninja", "einops", "transformers", "tilelang==0.1.8", "apache-tvm-ffi<=0.1.9",
        "quack-kernels>=0.3.4"]
missing = []
for r in reqs:
    name = re.split(r"[<>=]", r)[0]
    try:
        have = Version(md.version(name))
        spec = r[len(name):]
        if spec:
            from packaging.specifiers import SpecifierSet
            if have not in SpecifierSet(spec):
                missing.append(f"{r} (have {have})")
    except md.PackageNotFoundError:
        missing.append(r)
if missing:
    warnings.append("missing/mismatched runtime deps (install them, e.g. `pip install --no-deps "
                    + " ".join(f"'{m.split(' ')[0]}'" for m in missing) + "` plus their own deps): "
                    + ", ".join(missing))
for w in warnings:
    print("   WARNING:", w)
for e in errors:
    print("   ERROR:", e)
sys.exit(1 if errors else 0)
EOF
}

stage() {  # name repo tag
    local name=$1 repo=$2 tag=$3 dst="$BUILD_DIR/$1"
    rm -rf "$dst"
    if [[ "$MODE" == "--from-work" ]]; then
        echo ">> $name: copying work/$name (dev mode, patches not re-applied)"
        mkdir -p "$dst" && (cd "$ROOT/work/$name" && git ls-files -z | xargs -0 cp --parents -t "$dst")
    else
        echo ">> $name: fetching $tag"
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$tag" "$repo" "$dst"
        local patches=("$ROOT/patches/$name"/*.patch)
        if [[ -e "${patches[0]}" ]]; then
            echo ">> $name: applying ${#patches[@]} patch(es)"
            git -C "$dst" -c user.name=build -c user.email=build@localhost am -q --3way "${patches[@]}"
        fi
    fi
    if [[ -n "${CUDA_ARCHS:-}" ]]; then python3 "$ROOT/scripts/dev/limit_archs.py" "$dst/setup.py"; fi
}

[[ "$MODE" == "--patch-only" ]] || preflight
[[ "$MODE" == "--check" ]] && exit 0
mkdir -p "$BUILD_DIR"
stage causal-conv1d "$CAUSAL_CONV1D_REPO" "$CAUSAL_CONV1D_TAG"
stage mamba "$MAMBA_REPO" "$MAMBA_TAG"
[[ "$MODE" == "--patch-only" ]] && { echo "patched sources in $BUILD_DIR"; exit 0; }

export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
# --force-reinstall: the version string is unchanged across rebuilds, so without it pip may skip
#   the build and keep a .so compiled against a previous torch (-> undefined symbol at import).
# --no-deps: otherwise --force-reinstall also reinstalls torch & co. from PyPI, replacing your pins.
# --no-cache-dir: never reuse a cached wheel built against another torch.
PIP_FLAGS=(--no-build-isolation --force-reinstall --no-deps --no-cache-dir)
$PIP install "${PIP_FLAGS[@]}" "$BUILD_DIR/causal-conv1d"
$PIP install "${PIP_FLAGS[@]}" "$BUILD_DIR/mamba"

echo ">> smoke test"
(cd / && "$PYTHON" -c "
import torch, selective_scan_cuda, causal_conv1d_cuda, mamba_ssm
from mamba_ssm.utils.determinism import deterministic_configs  # present only in the patched build
print('   OK: mamba_ssm', mamba_ssm.__version__, 'on torch', torch.__version__)
")
echo ">> installed patched causal_conv1d ($CAUSAL_CONV1D_TAG) and mamba_ssm ($MAMBA_TAG)"
