#!/bin/bash
# Round 2: deeper encoder (e_layers=4) where HC-type connections should matter more.
mkdir -p logs/hc_exp
modes=(res hc mhc ohc)
preds=(96 192)
i=0
for p in "${preds[@]}"; do
  for m in "${modes[@]}"; do
    gpu=$((i % 8))
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2d4_96_${p}_${m} \
      --model iTransformerHC \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $p \
      --e_layers 4 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --hc_mode $m \
      --hc_expand 4 \
      --itr 1 > logs/hc_exp/ETTh2d4_96_${p}_${m}.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo "ALL DONE"
for f in logs/hc_exp/ETTh2d4_*.log; do echo "$f: $(grep -h '^mse:' $f | tail -1)"; done
