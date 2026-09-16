#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data custom \
  --root_path ../gapbench/csv \
  --data_path SZ_TAXI_15T.csv \
  --model_path ../models/chronos-bolt-small \
  --aux_head none \
  --task_loss regionfocal \
  --rf_au 2.0 \
  --epochs 10 \
  --patience 3 \
  --batch_size 32 \
  --lr 1e-5 \
  --num_workers 2 \
  --run_name boltft_rfau2_sztaxi
