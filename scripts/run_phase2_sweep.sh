#!/usr/bin/env bash
# Phase 2: hypothesis-verification sweep.
#
# For each MVSplat variant (full / base / w/o cost volume / w/o cross-view attn),
# evaluate on the local RE10K subset with:
#   - clean context views (reference)
#   - DeepJSCC-transmitted context views (mse & lpips variants) x SNR sweep
#
# All MVSplat weights stay frozen (mode=test). Results land in
# outputs/test/phase2/<run_name>/scores_all_avg.json; collect with
#   python scripts/collect_phase2_results.py
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase2_sweep.sh

set -e
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

PY=${PY:-python}
SNRS=${SNRS:-"0 5 10 15 20"}
JSCC_VARIANTS=${JSCC_VARIANTS:-"mse lpips"}
DATA_ROOT=${DATA_ROOT:-datasets/re10k}   # override to datasets/re10k_big
JSCC_DIR=${JSCC_DIR:-checkpoints/jscc}    # where jscc_<variant>_c12.pt live
OUT=${OUT:-phase2}                        # output subdir under outputs/test/

# name | checkpoint | extra hydra flags
MODELS=(
  "full|checkpoints/re10k.ckpt|"
  "base|checkpoints/ablations/re10k_worefine.ckpt|model.encoder.wo_depth_refine=true"
  "wocv|checkpoints/ablations/re10k_worefine_wocv.ckpt|model.encoder.wo_depth_refine=true model.encoder.wo_cost_volume=true"
  "wobbcrossattn|checkpoints/ablations/re10k_worefine_wobbcrossattn_best.ckpt|model.encoder.wo_depth_refine=true model.encoder.wo_backbone_cross_attn=true"
)

run_eval () {
  local name=$1 ckpt=$2 extra=$3 semcom=$4
  if [ -f "outputs/test/${OUT}/${name}/scores_all_avg.json" ]; then
    echo "[skip] ${name} (already done)"
    return
  fi
  echo "[run ] ${name}"
  $PY -m src.main +experiment=re10k \
    checkpointing.load="$ckpt" \
    mode=test dataset/view_sampler=evaluation \
    dataset.roots="[${DATA_ROOT}]" \
    test.compute_scores=true test.save_image=false \
    wandb.name="${OUT}/${name}" \
    $extra $semcom > "outputs/test/${OUT}_${name}.log" 2>&1
  tail -5 "outputs/test/${OUT}_${name}.log" | head -4
}

mkdir -p outputs/test

for entry in "${MODELS[@]}"; do
  IFS='|' read -r mname ckpt extra <<< "$entry"

  run_eval "${mname}_clean" "$ckpt" "$extra" ""

  for variant in $JSCC_VARIANTS; do
    weights="${JSCC_DIR}/jscc_${variant}_c12.pt"
    if [ ! -f "$weights" ]; then
      echo "[warn] $weights not found, skipping variant ${variant}"
      continue
    fi
    for snr in $SNRS; do
      run_eval "${mname}_${variant}_snr${snr}" "$ckpt" "$extra" \
        "+semcom=deepjscc semcom.weights=${weights} semcom.snr_db=${snr}.0"
    done
  done
done

echo "sweep complete."
