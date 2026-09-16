#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96_descw01 \
  --model PatchSTDesc \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp_descw01' \
  --itr 1 \
  --n_heads 4 \
  --train_epochs 3 --aux_head desc --aux_weight 0.1
