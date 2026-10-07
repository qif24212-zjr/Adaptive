#!/usr/bin/env bash
# CapERA PickNet-style sequential frame selection training.
# Must run with cwd = third_party/xmodaler (kfg.TEMP_DIR and config paths depend on it).
set -e

cd /root/autodl-tmp/uav_adaptive_captioning/third_party/xmodaler
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uavcap

python ../../scripts/train_capera.py \
    --num-gpus 1 \
    --config-file ../../configs/capera/picknet_style.yaml \
    "$@"
