#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data custom \
  --root_path ../gapbench/csv \
  --data_path SZ_TAXI_15T.csv \
  --model_path ../models/chronos-bolt-small \
  --aux_head none \
  --epochs 10 \
  --patience 3 \
  --batch_size 32 \
  --lr 1e-5 \
  --num_workers 2 \
  --run_name boltft_none_sztaxi
