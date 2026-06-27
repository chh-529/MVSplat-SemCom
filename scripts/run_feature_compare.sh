#!/usr/bin/env bash
# Finalizer: wait for the feature-JSCC training to finish, then evaluate the
# traditional feature codec (PCA+quant, no training) across SNR and plot the
# feature-domain comparison. Launch detached:
#   setsid nohup bash scripts/run_feature_compare.sh >outputs/feature_compare.log 2>&1 </dev/null &

export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=python
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=datasets/re10k_big
EVAL_SNRS="-10 -7 -5 -3 0 5 10 15 20"
log () { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for FEATJSCC_DONE..."
while ! grep -q FEATJSCC_DONE outputs/featjscc.log 2>/dev/null; do sleep 60; done
log "feature-JSCC done; evaluating traditional feature codec"

# Traditional feature codec needs no training: re10k.ckpt + fixed PCA codec.
CUDA_VISIBLE_DEVICES=0 SNRS="$EVAL_SNRS" DATA_ROOT=$DATA OUT=featjscc SEMCOM=feature_quant \
  bash scripts/run_phase3_eval.sh checkpoints/re10k.ckpt quant > outputs/feature_quant_eval.log 2>&1
log "traditional codec eval done"

PYTHONPATH="$(pwd)" $PY scripts/plot_feature_compare.py > outputs/feature_compare_table.log 2>&1
log "plot done"
cat outputs/feature_compare_table.log
log "FEATURE_COMPARE_DONE"
