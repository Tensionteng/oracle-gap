#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --num_workers 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/gapbench/csv/ \
  --data_path m4_weekly_W.csv \
  --model_id m4w_96_96_mae_s2023 \
  --model PatchTST \
  --data custom \
  --features M \
  --freq w \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 139 \
  --dec_in 139 \
  --c_out 139 \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1 --seed 2023 --task_loss mae
