#!/bin/bash
# Batch A: ETTh2 extreme depth. e_layers {32,64}, d_model=64, dropout=0.2, 5 arms x pred {96,192}.
mkdir -p logs/hc_exp
i=0
for el in 32 64; do
  for p in 96 192; do
    for m in res hc mhc ohc; do
      gpu=$((i % 8)); i=$((i+1))
      CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
        --model_id ETTh2e${el}d64_96_${p}_${m} --model iTransformerHC --data ETTh2 \
        --features M --seq_len 96 --label_len 48 --pred_len $p \
        --e_layers $el --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
        --des Exp --d_model 64 --d_ff 64 --dropout 0.2 --hc_mode $m --hc_expand 4 --itr 1 \
        > logs/hc_exp/ETTh2e${el}d64_96_${p}_${m}.log 2>&1 &
    done
    gpu=$((i % 8)); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --root_path ./dataset/ETT-small/ --data_path ETTh2.csv \
      --model_id ETTh2e${el}d64_96_${p}_respre --model iTransformerHC --data ETTh2 \
      --features M --seq_len 96 --label_len 48 --pred_len $p \
      --e_layers $el --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 \
      --des Exp --d_model 64 --d_ff 64 --dropout 0.2 --hc_mode hc --hc_expand 1 --itr 1 \
      > logs/hc_exp/ETTh2e${el}d64_96_${p}_respre.log 2>&1 &
  done
done
wait
echo "BATCH A DONE"
for f in logs/hc_exp/ETTh2e32d64_*.log logs/hc_exp/ETTh2e64d64_*.log; do
  echo "$f: $(grep -h '^mse:' $f | tail -1)"
done
