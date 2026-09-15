#!/bin/bash
# Confounder-controlled lookback sweep (iTransformer, pred_len=96).
# Every seq_len arm of a dataset trains on the SAME number of windows:
#   N_min = train windows available at seq_len=2880 (the longest lookback,
#           hence fewest windows), pred_len=96:
#     ETTh1:         8640 - 2880 - 96 + 1 = 5665
#     ETTm1:        34560 - 2880 - 96 + 1 = 31585
#     exchange_rate: int(7588*0.7)=5311 -> 5311 - 2880 - 96 + 1 = 2336
#     weather:       int(52696*0.7)=36887 -> 33912
#     electricity:   int(26304*0.7)=18412 -> 15437
#     traffic:       int(17544*0.7)=12280 -> 9305
# Shorter-lookback arms are evenly subsampled (deterministic, no RNG) inside
# data_provider/data_loader.py via --max_train_windows; val/test stay intact.
mkdir -p logs/lookback_ctrl
PY=.venv/bin/python

i=0
run() { # name data root csv enc_in d_model d_ff e_layers lr bsz seq_len n_min
  local name=$1 data=$2 root=$3 csv=$4 enc=$5 dm=$6 dff=$7 el=$8 lr=$9 bsz=${10} sl=${11} nmin=${12}
  local gpu=$((i % 8)); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --root_path $root --data_path $csv --model_id LBctrl_${name}_${sl} --model iTransformer \
    --data $data --features M --seq_len $sl --label_len 48 --pred_len 96 \
    --e_layers $el --d_layers 1 --factor 3 --enc_in $enc --dec_in $enc --c_out $enc \
    --des Exp --d_model $dm --d_ff $dff --learning_rate $lr --batch_size $bsz --itr 1 \
    --max_train_windows $nmin \
    > logs/lookback_ctrl/${name}_sl${sl}.log 2>&1 &
}

for sl in 96 336 720 1440 2880; do
  run ETTh1 ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 128 128 2 0.0001 32 $sl 5665
  run ETTm1 ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 128 128 2 0.0001 32 $sl 31585
  run exchange_rate custom ./dataset/exchange_rate/ exchange_rate.csv 8 128 128 2 0.0001 32 $sl 2336
  run weather custom ./dataset/weather/ weather.csv 21 512 512 3 0.0001 32 $sl 33912
  run electricity custom ./dataset/electricity/ electricity.csv 321 512 512 3 0.0005 16 $sl 15437
  run traffic custom ./dataset/traffic/ traffic.csv 862 512 512 3 0.0005 16 $sl 9305
done
wait
echo ALL DONE
