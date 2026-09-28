#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --num_workers 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/gapbench/csv/ \
  --data_path solar_H.csv \
  --model_id solar_96_96_rfau1_s2022 \
  --model PatchTST \
  --data custom \
  --features M \
  --freq h \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp_rf' \
  --batch_size 32 \
  --itr 1 --seed 2022 \
  --task_loss regionfocal \
  --rf_au 1.0
