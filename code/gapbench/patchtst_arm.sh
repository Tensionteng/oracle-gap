#!/usr/bin/env bash
# PatchTST arm driver: train each eligible dataset in TSLib (custom protocol),
# collect the tail-window dump, compute the gap. Run on a free GPU:
#   CUDA_VISIBLE_DEVICES=5 bash patchtst_arm.sh
set -euo pipefail
HERE=${MTP4TS_ROOT}/gapbench
TSLIB=${MTP4TS_ROOT}/Time-Series-Library
cd "$HERE"

declare -A CH=( [solar_H]=137 [SZ_TAXI_15T]=156 [LOOP_SEATTLE_H]=323 [M_DENSE_H]=30 \
                [ett1_H]=7 [ett2_H]=7 [jena_weather_H]=21 [bizitobs_l2c_H]=7 \
                [hierarchical_sales_D]=118 \
                [LOOP_SEATTLE_5T]=323 [solar_10T]=137 [ett1_15T]=7 [ett2_15T]=7 \
                [jena_weather_10T]=21 [bizitobs_l2c_5T]=7 )
declare -A FQ=( [solar_H]=h [SZ_TAXI_15T]=t [LOOP_SEATTLE_H]=h [M_DENSE_H]=h \
                [ett1_H]=h [ett2_H]=h [jena_weather_H]=h [bizitobs_l2c_H]=h \
                [hierarchical_sales_D]=d \
                [LOOP_SEATTLE_5T]=t [solar_10T]=t [ett1_15T]=t [ett2_15T]=t \
                [jena_weather_10T]=t [bizitobs_l2c_5T]=t )
# single-item multivariate datasets use strided tail windows (windows_full
# multi-window mode); multi-series collections use the single tail window
STRIDED="ett1_H ett2_H jena_weather_H bizitobs_l2c_H ett1_15T ett2_15T jena_weather_10T bizitobs_l2c_5T"

for key in "$@"; do
  ch=${CH[$key]}; fq=${FQ[$key]}
  echo "=== $key (channels=$ch freq=$fq) ==="
  cd "$TSLIB"
  uv run python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path "$HERE/csv/" --data_path "${key}.csv" \
    --model_id "gapbenchfull_${key}_96_96" \
    --model PatchTST --data custom --features M \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --e_layers 1 --d_layers 1 --factor 3 \
    --enc_in "$ch" --dec_in "$ch" --c_out "$ch" \
    --des Exp --n_heads 2 --itr 1 --freq "$fq" --num_workers 4 \
    > "$HERE/logs/patchtst_${key}.log" 2>&1
  cd "$HERE"
  if [[ " $STRIDED " == *" $key "* ]]; then
    uv run --project ../tsfm_stage2 python collect_tslib_dump.py \
      --model_id "gapbenchfull_${key}_96_96" --csv "csv/${key}.csv" \
      --out "dumps_full/${key}_tail/patchtst" --tail_strided 20
  else
    uv run --project ../tsfm_stage2 python collect_tslib_dump.py \
      --model_id "gapbenchfull_${key}_96_96" --csv "csv/${key}.csv" \
      --out "dumps_full/${key}_tail/patchtst" --tail_only
  fi
done
echo ALL DONE
