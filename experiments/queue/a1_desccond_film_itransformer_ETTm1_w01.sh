#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/Time-Series-Library
uv run python -u run.py \
  --num_workers 2 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_condf \
  --model iTransformerDesc \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --des 'Exp_condf' \
  --aux_head desc \
  --desc_cond film \
  --aux_weight 0.1 \
  --itr 1
