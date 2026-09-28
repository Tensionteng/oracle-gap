#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --model_path ../models/chronos-bolt-small \
  --aux_head none \
  --epochs 10 \
  --patience 3 \
  --batch_size 32 \
  --lr 1e-5 \
  --num_workers 2 \
  --run_name tsfm_bolts_none_ETTh1
