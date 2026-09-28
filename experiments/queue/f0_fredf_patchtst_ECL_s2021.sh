#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --num_workers 3 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_fredf \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp_fredf' \
  --batch_size 16 \
  --task_loss fredf \
  --itr 1
