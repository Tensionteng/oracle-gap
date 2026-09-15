#!/bin/bash
# Batch B: ECL (321 variates, wide). Official iTransformer hyperparams, e_layers {3,6}, 5 arms x pred {96,192}.
mkdir -p logs/hc_exp
i=0
for el in 3 6; do
  for p in 96 192; do
    for m in res hc mhc ohc; do
      gpu=$((i % 8)); i=$((i+1))
      CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path ./dataset/electricity/ --data_path electricity.csv \
        --model_id ECLe${el}_96_${p}_${m} --model iTransformerHC --data custom \
        --features M --seq_len 96 --label_len 48 --pred_len $p \
        --e_layers $el --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 \
        --des Exp --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.0005 \
        --hc_mode $m --hc_expand 4 --itr 1 \
        > logs/hc_exp/ECLe${el}_96_${p}_${m}.log 2>&1 &
    done
    gpu=$((i % 8)); i=$((i+1))
    CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
      --task_name long_term_forecast --is_training 1 \
      --root_path ./dataset/electricity/ --data_path electricity.csv \
      --model_id ECLe${el}_96_${p}_respre --model iTransformerHC --data custom \
      --features M --seq_len 96 --label_len 48 --pred_len $p \
      --e_layers $el --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 \
      --des Exp --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.0005 \
      --hc_mode hc --hc_expand 1 --itr 1 \
      > logs/hc_exp/ECLe${el}_96_${p}_respre.log 2>&1 &
  done
done
wait
echo "BATCH B DONE"
for f in logs/hc_exp/ECLe3_*.log logs/hc_exp/ECLe6_*.log; do
  echo "$f: $(grep -h '^mse:' $f | tail -1)"
done
