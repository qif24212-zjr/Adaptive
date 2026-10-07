#!/usr/bin/env bash
# CapERA PickNet-style evaluation (--eval-only).
# Prints BLEU-4 / METEOR / ROUGE-L / CIDEr + selection summary and dumps the
# selection trace JSON to experiments/capera_picknet_style/results/.
# Usage: bash scripts/eval_capera_picknet.sh [path/to/checkpoint.pth]
set -e

cd /root/autodl-tmp/uav_adaptive_captioning/third_party/xmodaler
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uavcap

CKPT="${1:-/root/autodl-tmp/uav_adaptive_captioning/experiments/capera_picknet_style/model_final.pth}"

python ../../scripts/train_capera.py \
    --num-gpus 1 \
    --eval-only \
    --config-file ../../configs/capera/picknet_style.yaml \
    MODEL.WEIGHTS "$CKPT"
