#!/usr/bin/env bash
# CapERA XE baseline evaluation (--eval-only). Prints BLEU-4 / METEOR / ROUGE-L / CIDEr.
# Usage: bash scripts/eval_capera.sh [path/to/checkpoint.pth]
set -e

cd /root/autodl-tmp/uav_adaptive_captioning/third_party/xmodaler
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uavcap

CKPT="${1:-/root/autodl-tmp/uav_adaptive_captioning/experiments/capera_xe_baseline/model_final.pth}"

python ../../scripts/train_capera.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ../../configs/capera/xe_baseline.yaml \
    MODEL.WEIGHTS "$CKPT"
