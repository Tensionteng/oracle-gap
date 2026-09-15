#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data ETTm1 \
  --data_path ETTm1.csv \
  --model_path ../models/chronos-bolt-small \
  --eval_only \
  --num_workers 2 \
  --run_name tsfm_bolts_zeroshot_ETTm1
