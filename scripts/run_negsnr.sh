#!/usr/bin/env bash
# Negative-SNR study: retrain JSCC over SNR [-10,20] so it is valid in the cliff
# region, then compare DeepJSCC (frozen + e2e) against JPEG+ideal-coding across
# SNR -10..20 on the 500-scene test set. Shows JPEG's cliff vs JSCC's graceful
# degradation. Every stage is idempotent.
# Launch: setsid nohup bash scripts/run_negsnr.sh >outputs/negsnr.log 2>&1 </dev/null &

export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DATA=datasets/re10k_big
JSCC_NEG=checkpoints/jscc_neg
EVAL_SNRS="-10 -7 -5 -3 0 5 10 15 20"
mkdir -p "$JSCC_NEG" outputs/phase3_neg
log () { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Stage 1: JSCC mse pretrain [-10,20] (GPU0) || JPEG sweep big test (GPU1) ----
log "Stage 1: JSCC retrain [-10,20] || JPEG sweep"
(
  if [ ! -f "$JSCC_NEG/jscc_mse_c12.pt" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY -m src.scripts.train_deepjscc --variant mse --steps 40000 \
      --snr_min -10 --snr_max 20 --data_root $DATA --out_dir $JSCC_NEG \
      --max_frames 200000 --num_workers 6 > outputs/negsnr_jscc_mse.log 2>&1
    log "  JSCC retrain done"
  fi
) &
(
  CUDA_VISIBLE_DEVICES=1 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=jpeg_big \
    bash scripts/run_jpeg_compare.sh > outputs/negsnr_jpeg.log 2>&1
  log "  JPEG sweep done"
) &
wait
log "Stage 1 complete"

# ---- Stage 2: frozen JSCC eval (GPU0) || e2e fine-tune [-10,20] (GPU1) ----
log "Stage 2: frozen eval || e2e [-10,20] train"
(
  CUDA_VISIBLE_DEVICES=0 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=negsnr \
    SEMCOM_EXTRA="semcom.weights=$JSCC_NEG/jscc_mse_c12.pt" \
    bash scripts/run_phase3_eval.sh checkpoints/re10k.ckpt frozen > outputs/negsnr_eval_frozen.log 2>&1
  log "  frozen eval done"
) &
(
  if ! ls outputs/phase3_neg/e2e/checkpoints/*step_20000* >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=1 $PY -m src.main +experiment=re10k +semcom=deepjscc \
      checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false checkpointing.every_n_train_steps=5000 \
      semcom.weights=$JSCC_NEG/jscc_mse_c12.pt semcom.trainable=true semcom.snr_db=null \
      semcom.snr_min=-10.0 semcom.snr_max=20.0 \
      optimizer.lr=2e-5 optimizer.semcom_lr=1e-4 \
      dataset.roots=[$DATA] data_loader.train.batch_size=2 trainer.max_steps=20000 \
      trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 wandb.mode=disabled \
      output_dir=outputs/phase3_neg/e2e > outputs/negsnr_train_e2e.log 2>&1
    log "  e2e train done"
  fi
) &
wait
log "Stage 2 complete"

# ---- Stage 3: e2e JSCC eval across SNR ----
log "Stage 3: e2e eval"
ECKPT=$(ls outputs/phase3_neg/e2e/checkpoints/*step_20000* 2>/dev/null | head -1)
[ -n "$ECKPT" ] && CUDA_VISIBLE_DEVICES=0 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=negsnr \
  bash scripts/run_phase3_eval.sh "$ECKPT" e2e > outputs/negsnr_eval_e2e.log 2>&1
log "Stage 3 complete"
log "NEGSNR_DONE"
