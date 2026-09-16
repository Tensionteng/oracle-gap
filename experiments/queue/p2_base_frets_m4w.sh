#!/usr/bin/env bash
set -euo pipefail
cd ${MTP4TS_ROOT}/Time-Series-Library
uv run python -u run.py   --task_name long_term_forecast   --is_training 1   --root_path ./gapbench/csv/   --data_path m4_weekly_W.csv   --model_id m4w_96_96_base_frets
