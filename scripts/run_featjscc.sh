#!/usr/bin/env bash
# Feature-domain SemCom (insertion point B): train + eval, compare to pixel-domain.
# Stage 1: (b) train feature-JSCC only (GPU0) || (c) full e2e (GPU1), on re10k_big.
# Stage 2: eval (b) and (c) across SNR on the 429-scene test set.
# Feature JSCC has no natural pretraining, so it is trained from scratch with the
# rendering loss (task-oriented). Same R=1/8 as the pixel-domain JSCC.
# Launch: setsid nohup bash scripts/run_featjscc.sh >outputs/featjscc.log 2>&1 </dev/null &

export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DATA=datasets/re10k_big
EVAL_SNRS="-10 -7 -5 -3 0 5 10 15 20"
mkdir -p outputs/featjscc
log () { echo "[$(date +%H:%M:%S)] $*"; }

COMMON="+experiment=re10k +semcom=feature_jscc checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false checkpointing.every_n_train_steps=5000 \
  semcom.trainable=true semcom.snr_db=null semcom.snr_min=-10.0 semcom.snr_max=20.0 \
  dataset.roots=[$DATA] data_loader.train.batch_size=2 trainer.max_steps=20000 \
  trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 wandb.mode=disabled"

# ---- Stage 1: train ----
log "Stage 1: train feature-JSCC (b) || full e2e (c)"
(
  if ! ls outputs/featjscc/b_featonly/checkpoints/*step_20000* >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=0 $PY -m src.main $COMMON \
      optimizer.freeze_mvsplat=true optimizer.semcom_lr=1e-4 \
      output_dir=outputs/featjscc/b_featonly > outputs/featjscc_train_b.log 2>&1
    log "  (b) done"
  fi
) &
(
  if ! ls outputs/featjscc/c_e2e/checkpoints/*step_20000* >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=1 $PY -m src.main $COMMON \
      optimizer.lr=2e-5 optimizer.semcom_lr=1e-4 \
      output_dir=outputs/featjscc/c_e2e > outputs/featjscc_train_c.log 2>&1
    log "  (c) done"
  fi
) &
wait
log "Stage 1 complete"

# ---- Stage 2: eval across SNR ----
log "Stage 2: eval"
BCKPT=$(ls outputs/featjscc/b_featonly/checkpoints/*step_20000* 2>/dev/null | head -1)
CCKPT=$(ls outputs/featjscc/c_e2e/checkpoints/*step_20000* 2>/dev/null | head -1)
(
  [ -n "$BCKPT" ] && CUDA_VISIBLE_DEVICES=0 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=featjscc SEMCOM=feature_jscc \
    bash scripts/run_phase3_eval.sh "$BCKPT" featonly > outputs/featjscc_eval_b.log 2>&1
  log "  eval (b) done"
) &
(
  [ -n "$CCKPT" ] && CUDA_VISIBLE_DEVICES=1 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=featjscc SEMCOM=feature_jscc \
    bash scripts/run_phase3_eval.sh "$CCKPT" e2e > outputs/featjscc_eval_c.log 2>&1
  log "  eval (c) done"
) &
wait
log "Stage 2 complete"
log "FEATJSCC_DONE"
