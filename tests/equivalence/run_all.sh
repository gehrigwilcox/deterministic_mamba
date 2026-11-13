#!/usr/bin/env bash
# Usage: run_all.sh STOCK_PYTHON PATCHED_PYTHON OUTDIR [steps]
set -euo pipefail
STOCK=$1; PATCHED=$2; OUT=$3; STEPS=${4:-300}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset MAMBA_DETERMINISTIC CAUSAL_CONV1D_DETERMINISTIC
mkdir -p "$OUT"
cd "$OUT"  # don't let a source checkout shadow the installed packages
# sanity: the two interpreters must load different builds
$STOCK   -c "import mamba_ssm.utils.determinism as d; assert not hasattr(d, 'deterministic_configs'), 'stock env loads patched mamba_ssm'; print('stock  :', d.__file__)"
$PATCHED -c "import mamba_ssm.utils.determinism as d; assert hasattr(d, 'deterministic_configs'), 'patched env loads stock mamba_ssm'; print('patched:', d.__file__)"

if [[ -z "${SKIP_KERNELS:-}" ]]; then
echo "=== kernel / module outputs"
$STOCK   "$HERE/kernel_outputs.py" k_stock1.pt default
$STOCK   "$HERE/kernel_outputs.py" k_stock2.pt default
$PATCHED "$HERE/kernel_outputs.py" k_pdef.pt default
$PATCHED "$HERE/kernel_outputs.py" k_pdet.pt det
$PATCHED "$HERE/compare.py" k_stock1.pt k_stock2.pt k_pdef.pt k_pdet.pt | tee kernel_report.txt
fi

echo "=== training curves ($STEPS steps)"
for layer in Mamba1 Mamba2 Mamba3; do
    $STOCK   "$HERE/train_curve.py" t_${layer}_stock1.pt $layer default $STEPS
    $STOCK   "$HERE/train_curve.py" t_${layer}_stock2.pt $layer default $STEPS
    $PATCHED "$HERE/train_curve.py" t_${layer}_pdef.pt   $layer default $STEPS
    $PATCHED "$HERE/train_curve.py" t_${layer}_pdet.pt   $layer det     $STEPS
done
for layer in Mamba1 Mamba2 Mamba3; do
    echo "--- $layer"
    $PATCHED "$HERE/compare_curves.py" t_${layer}_stock1.pt t_${layer}_stock2.pt t_${layer}_pdef.pt t_${layer}_pdet.pt
done | tee curve_report.txt
