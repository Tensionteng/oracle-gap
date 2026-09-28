#!/usr/bin/env bash
set -euo pipefail
cd /mnt/jd/users/tengshiyuan.1/codes/mtp4ts/tsfm_stage2
export HF_HUB_OFFLINE=1
uv run python -u finetune_chronos_bolt.py \
  --data custom \
  --root_path ../Time-Series-Library/dataset/traffic \
  --data_path traffic.csv \
  --model_path ../models/chronos-bolt-small \
  --aux_head desc \
  --aux_weight 0.1 \
  --epochs 10 \
  --patience 3 \
  --batch_size 32 \
  --lr 1e-5 \
  --num_workers 2 \
  --run_name tsfm_bolts_descw01_traffic
uv run python -u errcorr_bolt.py \
  --data custom \
  --root_path ../Time-Series-Library/dataset/traffic \
  --data_path traffic.csv \
  --ckpt checkpoints/tsfm_bolts_descw01_traffic \
  --dump pred_dumps/tsfm_bolts_descw01_traffic \
  --device cuda \
  --out results/errcorr_bolt_traffic.json
