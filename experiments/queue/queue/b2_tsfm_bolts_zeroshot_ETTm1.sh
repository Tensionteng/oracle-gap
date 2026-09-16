#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data ETTm1 \
  --data_path ETTm1.csv \
  --model_path ../models/chronos-bolt-small \
  --eval_only \
  --num_workers 2 \
  --run_name tsfm_bolts_zeroshot_ETTm1
