#!/usr/bin/env bash
# CapERA uniform-frame XE baseline training.
# Must run with cwd = third_party/xmodaler (kfg.TEMP_DIR and config paths depend on it).
set -e

cd /root/autodl-tmp/uav_adaptive_captioning/third_party/xmodaler
source /root/miniconda3/etc/profile.d/conda.sh
conda activate uavcap

python ../../scripts/train_capera.py \
    --num-gpus 1 \
    --config-file ../../configs/capera/xe_baseline.yaml \
    "$@"
