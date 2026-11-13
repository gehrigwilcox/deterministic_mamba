#!/usr/bin/env bash
# Fetch the pinned upstream tags, apply patches/, and pip-install the patched packages
# (they keep their upstream names: `causal_conv1d` and `mamba_ssm`).
#
#   scripts/build.sh                 # fetch + patch + install
#   scripts/build.sh --patch-only    # fetch + patch into $BUILD_DIR, don't install
#   scripts/build.sh --from-work     # dev: build from work/<repo> instead of fetching
#
# Env:
#   CUDA_ARCHS="80;90"   only compile these SM architectures (default: upstream's full list)
#   BUILD_DIR=...        where sources are staged (default: ./build)
#   PIP="python -m pip"  pip to install with (use the interpreter of your training env)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/versions.env"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PIP="${PIP:-python -m pip}"
MODE="${1:-install}"

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

mkdir -p "$BUILD_DIR"
stage causal-conv1d "$CAUSAL_CONV1D_REPO" "$CAUSAL_CONV1D_TAG"
stage mamba "$MAMBA_REPO" "$MAMBA_TAG"
[[ "$MODE" == "--patch-only" ]] && { echo "patched sources in $BUILD_DIR"; exit 0; }

export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
$PIP install --no-build-isolation "$BUILD_DIR/causal-conv1d"
$PIP install --no-build-isolation "$BUILD_DIR/mamba"
echo ">> installed patched causal_conv1d ($CAUSAL_CONV1D_TAG) and mamba_ssm ($MAMBA_TAG)"
