#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_descw03 \
  --model PatchSTDesc \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 2 \
  --des 'Exp_descw03' \
  --aux_head desc \
  --aux_weight 0.3 \
  --itr 1
