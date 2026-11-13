#!/usr/bin/env bash
# Regenerate patches/<repo>/*.patch from the commits on top of the pinned tag in work/<repo>.
# Workflow for a new upstream version: clone the new tag into work/<repo>, `git am` the old
# patches (resolve conflicts), run the tests, then run this script and bump versions.env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/versions.env"
export_repo() {  # name tag
    local name=$1 tag=$2
    rm -f "$ROOT/patches/$name"/*.patch
    mkdir -p "$ROOT/patches/$name"
    git -C "$ROOT/work/$name" format-patch -q --no-signature --zero-commit -o "$ROOT/patches/$name" "$tag"
    ls "$ROOT/patches/$name"
}
export_repo causal-conv1d "$CAUSAL_CONV1D_TAG"
export_repo mamba "$MAMBA_TAG"
