#!/usr/bin/env bash
# 汇总已完成任务的最后一条 mse/mae，输出 experiments/results.csv
set -u
ROOT=${MTP4TS_ROOT}
OUT=$ROOT/experiments/results.csv
echo "job,mse,mae" > "$OUT"
for d in "$ROOT"/experiments/done/*.done; do
  [ -e "$d" ] || continue
  name=$(basename "$d" .done)
  line=$(grep -aoE "mse:[0-9.]+, mae:[0-9.]+" "$ROOT/experiments/logs/$name.log" | tail -1)
  if [ -n "$line" ]; then
    mse=$(echo "$line" | sed -E 's/mse:([0-9.]+),.*/\1/')
    mae=$(echo "$line" | sed -E 's/.*mae:([0-9.]+)/\1/')
    echo "$name,$mse,$mae" >> "$OUT"
  else
    echo "$name,FAILED,FAILED" >> "$OUT"
  fi
done
echo "wrote $OUT"
