#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py \
  --num_workers 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ${MTP4TS_ROOT}/gapbench/csv/ \
  --data_path us_births_W.csv \
  --model_id usbirths_96_96_rfau1_s2022 \
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
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp_rf' \
  --batch_size 16 \
  --itr 1 --seed 2022 \
  --task_loss regionfocal \
  --rf_au 1.0
