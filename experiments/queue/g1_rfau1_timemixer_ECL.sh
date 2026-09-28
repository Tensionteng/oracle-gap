#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id rfau1_ECL_96'_'96 \
  --model TimeMixer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 --task_loss regionfocal --rf_au 1.0 --num_workers 2 \
  --d_model 16 \
  --d_ff 32 \
  --batch_size 32 \
  --learning_rate 0.01 \
  --train_epochs 20 \
  --patience 10 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2
