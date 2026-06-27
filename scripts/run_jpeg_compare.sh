#!/usr/bin/env bash
# SemCom (DeepJSCC) vs classical JPEG + ideal channel coding.
#
# At each SNR the classical system gets the capacity bit budget for our bandwidth
# (R = 1/8, 256x256 -> k = 24576 complex uses): bpp = 0.375 * log2(1 + SNR_lin).
# JPEG is compressed to that budget; below the budget-feasible point the link is
# in outage (gray frame) -> the classical cliff effect. This is the steel-man for
# JPEG (ideal capacity-achieving code); real codes cliff at higher SNR.
#
# JSCC numbers for the overlay come from the Phase-2 full-model sweep (SNR 0..20).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_jpeg_compare.sh

set -e
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
PY=${PY:-python}
CKPT=checkpoints/re10k.ckpt
SNRS=${SNRS:-"-10 -5 0 5 10 15 20"}
DATA_ROOT=${DATA_ROOT:-datasets/re10k}   # override to datasets/re10k_big
OUT=${OUT:-jpeg}                          # output subdir under outputs/test/

run () {
  local name=$1 semcom=$2
  if [ -f "outputs/test/${OUT}/${name}/scores_all_avg.json" ]; then echo "[skip] ${name}"; return; fi
  echo "[run ] ${name}"
  $PY -m src.main +experiment=re10k checkpointing.load="$CKPT" \
    mode=test dataset/view_sampler=evaluation \
    dataset.roots="[${DATA_ROOT}]" \
    test.compute_scores=true test.save_image=false \
    wandb.name="${OUT}/${name}" $semcom > "outputs/test/${OUT}_${name}.log" 2>&1
}

mkdir -p outputs/test
for snr in $SNRS; do
  bpp=$($PY -c "import math;print(round(0.375*math.log2(1+10**($snr/10)),4))")
  echo "SNR ${snr}dB -> bpp ${bpp}"
  run "jpeg_snr${snr}" "+semcom=degradation semcom.kind=jpeg semcom.strength=${bpp}"
done
echo "jpeg compare sweep complete."
