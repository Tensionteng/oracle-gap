#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --num_workers 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/gapbench/csv/ \
  --data_path SZ_TAXI_15T.csv \
  --model_id SZ_TAXI_96_96_rfau2 \
  --model PatchTST \
  --data custom \
  --features M \
  --freq t \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 156 \
  --dec_in 156 \
  --c_out 156 \
  --des 'Exp_rf' \
  --batch_size 16 \
  --itr 1 \
  --task_loss regionfocal \
  --rf_au 2.0
