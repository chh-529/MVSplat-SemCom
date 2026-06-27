#!/usr/bin/env bash
# Overnight autonomous pipeline on the larger downloaded dataset (re10k_big):
#   0. wait for the download to reach enough scenes
#   1. (parallel) Phase-2 re-eval on 500 test  ||  JSCC retrain (mse+lpips) on big train
#   2. (parallel) Phase-3 retrain (b) train-JSCC-only || (c) full e2e, from the new JSCC
#   3. (parallel) Phase-3 eval: frozen baseline + (b) || (c), on big test
# Every stage is idempotent (skips finished work) so a restart resumes.
# Launch detached:  setsid nohup bash scripts/run_overnight.sh >outputs/overnight.log 2>&1 </dev/null &

export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=python
DATA=datasets/re10k_big
JSCC_BIG=checkpoints/jscc_big
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$JSCC_BIG" outputs/phase3_big

log () { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Stage 0: wait for the download ----
log "Stage 0: waiting for download (need >=2500 train scenes or done marker)"
for i in $(seq 1 240); do   # up to ~8h
  tr=$($PY -c "import json,os;p='$DATA/train/index.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)" 2>/dev/null)
  if grep -q ALL_DOWNLOAD_DONE outputs/download_re10k_big.log 2>/dev/null; then log "download finished (train=$tr)"; break; fi
  if [ "${tr:-0}" -ge 2500 ]; then log "enough train scenes ($tr), proceeding"; break; fi
  sleep 120
done

# ---- Stage 1: Phase-2 re-eval (GPU0, old JSCC) || JSCC retrain (GPU1) ----
log "Stage 1: phase2 re-eval || JSCC retrain"
(
  CUDA_VISIBLE_DEVICES=0 DATA_ROOT=$DATA JSCC_DIR=checkpoints/jscc OUT=phase2_big \
    bash scripts/run_phase2_sweep.sh > outputs/overnight_phase2_big.log 2>&1
  log "  phase2 re-eval done"
) &
(
  for v in mse lpips; do
    if [ ! -f "$JSCC_BIG/jscc_${v}_c12.pt" ]; then
      log "  JSCC retrain ($v) starting"
      CUDA_VISIBLE_DEVICES=1 $PY -m src.scripts.train_deepjscc --variant $v --steps 40000 \
        --data_root $DATA --out_dir $JSCC_BIG --max_frames 200000 --num_workers 6 \
        > outputs/overnight_jscc_${v}.log 2>&1
      log "  JSCC retrain ($v) done"
    fi
  done
) &
wait
log "Stage 1 complete"

# ---- Stage 2: Phase-3 retrain (b) GPU0 || (c) GPU1, from new jscc_mse ----
log "Stage 2: phase3 retrain"
COMMON="+experiment=re10k +semcom=deepjscc checkpointing.load=checkpoints/re10k.ckpt \
  checkpointing.resume=false checkpointing.every_n_train_steps=5000 \
  semcom.weights=$JSCC_BIG/jscc_mse_c12.pt semcom.trainable=true semcom.snr_db=null \
  dataset.roots=[$DATA] data_loader.train.batch_size=2 trainer.max_steps=20000 \
  trainer.limit_val_batches=0 trainer.num_sanity_val_steps=0 wandb.mode=disabled"
(
  if ! ls outputs/phase3_big/b_jscconly/checkpoints/*step_20000* >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=0 $PY -m src.main $COMMON \
      optimizer.freeze_mvsplat=true optimizer.semcom_lr=1e-4 \
      output_dir=outputs/phase3_big/b_jscconly > outputs/overnight_phase3_b.log 2>&1
    log "  phase3 (b) done"
  fi
) &
(
  if ! ls outputs/phase3_big/c_e2e/checkpoints/*step_20000* >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=1 $PY -m src.main $COMMON \
      optimizer.lr=2e-5 optimizer.semcom_lr=1e-4 \
      output_dir=outputs/phase3_big/c_e2e > outputs/overnight_phase3_c.log 2>&1
    log "  phase3 (c) done"
  fi
) &
wait
log "Stage 2 complete"

# ---- Stage 3: Phase-3 eval (frozen baseline + b on GPU0, c on GPU1) ----
log "Stage 3: phase3 eval"
BCKPT=$(ls outputs/phase3_big/b_jscconly/checkpoints/*step_20000* 2>/dev/null | head -1)
CCKPT=$(ls outputs/phase3_big/c_e2e/checkpoints/*step_20000* 2>/dev/null | head -1)
(
  CUDA_VISIBLE_DEVICES=0 DATA_ROOT=$DATA OUT=phase3_big \
    SEMCOM_EXTRA="semcom.weights=$JSCC_BIG/jscc_mse_c12.pt" \
    bash scripts/run_phase3_eval.sh checkpoints/re10k.ckpt frozen > outputs/overnight_eval_frozen.log 2>&1
  [ -n "$BCKPT" ] && CUDA_VISIBLE_DEVICES=0 DATA_ROOT=$DATA OUT=phase3_big \
    bash scripts/run_phase3_eval.sh "$BCKPT" jscconly > outputs/overnight_eval_b.log 2>&1
  log "  eval frozen+b done"
) &
(
  [ -n "$CCKPT" ] && CUDA_VISIBLE_DEVICES=1 DATA_ROOT=$DATA OUT=phase3_big \
    bash scripts/run_phase3_eval.sh "$CCKPT" e2e > outputs/overnight_eval_c.log 2>&1
  log "  eval c done"
) &
wait
log "Stage 3 complete"

log "OVERNIGHT_DONE"
