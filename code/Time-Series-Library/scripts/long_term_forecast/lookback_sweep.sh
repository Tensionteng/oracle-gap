#!/bin/bash
# Lookback sweep: does longer context help, and where does it roll over?
# 5 datasets x 5 seq_lens (iTransformer) + DLinear reference on ETTh1/ECL, pred_len=96 fixed.
mkdir -p logs/lookback
i=0
run() { # data root path csv enc_in d_model d_ff e_layers lr bsz seq
  local data=$1 root=$2 csv=$3 enc=$4 dm=$5 dff=$6 el=$7 lr=$8 bsz=$9 sl=${10} model=${11}
  local gpu=$((i % 8)); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path $root --data_path $csv --model_id LB_${data}_${sl} --model $model \
    --data $data --features M --seq_len $sl --label_len 48 --pred_len 96 \
    --e_layers $el --d_layers 1 --factor 3 --enc_in $enc --dec_in $enc --c_out $enc \
    --des Exp --d_model $dm --d_ff $dff --learning_rate $lr --batch_size $bsz --itr 1 \
    > logs/lookback/${data}_${model}_sl${sl}.log 2>&1 &
}
for sl in 96 336 720 1440 2880; do
  run ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 128 128 2 0.0001 32 $sl iTransformer
  run ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 128 128 2 0.0001 32 $sl iTransformer
  run exchange_rate ./dataset/exchange_rate/ exchange_rate.csv 8 128 128 2 0.0001 32 $sl iTransformer
  run weather ./dataset/weather/ weather.csv 21 512 512 3 0.0001 32 $sl iTransformer
  run custom ./dataset/electricity/ electricity.csv 321 512 512 3 0.0005 16 $sl iTransformer
done
# DLinear reference
for sl in 96 336 720 1440 2880; do
  run ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 128 128 2 0.0001 32 $sl DLinear
done
wait
echo ALL DONE
for f in logs/lookback/*.log; do echo "$(basename $f .log): $(grep -h '^mse:' $f | tail -1 | cut -d, -f1)"; done
