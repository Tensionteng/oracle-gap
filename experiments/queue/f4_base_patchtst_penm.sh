#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py \
  --num_workers 3 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${MTP4TS_ROOT}/data/censor/ \
  --data_path penm_fc.csv \
  --model_id penm_96_96 \
  --model PatchTST \
  --data custom \
  --features M \
  --target power \
  --freq t \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 5 \
  --des 'Exp' \
  --batch_size 16 \
  --itr 1
