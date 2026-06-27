#!/usr/bin/env bash
# Evaluate a Phase-3 fine-tuned checkpoint across SNR (full model).
# The checkpoint already contains both MVSplat and the trained SemCom weights,
# so semcom.weights stays null (the checkpoint provides them).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase3_eval.sh <ckpt> <tag>
#   e.g. ... outputs/phase3/c_e2e/checkpoints/epoch_*-step_20000.ckpt e2e

set -e
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
PY=${PY:-python}
CKPT=$1
TAG=$2
SNRS=${SNRS:-"0 5 10 15 20"}
DATA_ROOT=${DATA_ROOT:-datasets/re10k}   # override to datasets/re10k_big
OUT=${OUT:-phase3}
SEMCOM=${SEMCOM:-deepjscc}                # deepjscc (pixel) | feature_jscc (feature)
# Optional extra semcom flags, e.g. to eval a frozen baseline with given weights:
#   SEMCOM_EXTRA="semcom.weights=checkpoints/jscc_big/jscc_mse_c12.pt"
SEMCOM_EXTRA=${SEMCOM_EXTRA:-}
[ -z "$CKPT" ] && { echo "usage: $0 <ckpt> <tag>"; exit 1; }

for snr in $SNRS; do
  name="${TAG}_snr${snr}"
  [ -f "outputs/test/${OUT}/${name}/scores_all_avg.json" ] && { echo "[skip] $name"; continue; }
  echo "[run ] $name"
  $PY -m src.main +experiment=re10k +semcom=${SEMCOM} \
    checkpointing.load="$CKPT" semcom.snr_db=${snr}.0 $SEMCOM_EXTRA \
    mode=test dataset/view_sampler=evaluation \
    dataset.roots="[${DATA_ROOT}]" \
    test.compute_scores=true test.save_image=false \
    wandb.name="${OUT}/${name}" > "outputs/test/${OUT}_${name}.log" 2>&1
done
echo "phase3 eval ($TAG) complete."
