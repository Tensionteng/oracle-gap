#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py \
  --num_workers 3 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${MTP4TS_ROOT}/data/swat/ \
  --data_path swat.csv \
  --model_id swat_96_96_rfau05 \
  --model PatchTST \
  --data custom \
  --features M \
  --target P603 \
  --freq t \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --des 'Exp_rf05' \
  --batch_size 16 \
  --itr 1 --task_loss regionfocal --rf_au 0.5 --num_workers 2
