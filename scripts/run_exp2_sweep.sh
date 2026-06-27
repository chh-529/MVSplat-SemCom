#!/usr/bin/env bash
# Experiment 2: view-consistent vs view-independent distortion (mechanism test).
#
# Full MVSplat (frozen) evaluated on the local RE10K subset under matched-magnitude
# distortions whose cross-view consistency is toggled:
#   - noise/independent : per-view noise draw   (inconsistent)
#   - noise/shared      : shared noise draw     (consistent, same magnitude)
#   - blur              : same Gaussian blur all views (coherent reference)
#
# Each run self-reports its transmission PSNR (tx_psnr) + task PSNR, so we can
# plot task-vs-transmission curves and read off the consistency effect.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_exp2_sweep.sh

set -e
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
PY=${PY:-python}
CKPT=checkpoints/re10k.ckpt
NOISE_STD=${NOISE_STD:-"0.05 0.08 0.12 0.18 0.25"}
BLUR_SIGMA=${BLUR_SIGMA:-"0.5 1.0 1.5 2.5 4.0"}

run () {
  local name=$1 semcom=$2
  if [ -f "outputs/test/exp2/${name}/scores_all_avg.json" ]; then
    echo "[skip] ${name}"; return
  fi
  echo "[run ] ${name}"
  $PY -m src.main +experiment=re10k checkpointing.load="$CKPT" \
    mode=test dataset/view_sampler=evaluation \
    test.compute_scores=true test.save_image=false \
    wandb.name="exp2/${name}" $semcom > "outputs/test/exp2_${name}.log" 2>&1
}

mkdir -p outputs/test
for std in $NOISE_STD; do
  for corr in independent shared; do
    run "noise_${corr}_std${std}" \
      "+semcom=degradation semcom.kind=noise semcom.strength=${std} semcom.correlation=${corr}"
  done
done
for sig in $BLUR_SIGMA; do
  run "blur_sigma${sig}" "+semcom=degradation semcom.kind=blur semcom.strength=${sig}"
done
echo "exp2 sweep complete."
