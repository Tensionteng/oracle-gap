#!/usr/bin/env bash
# 实验队列调度器：按字典序执行 experiments/queue/*.sh，一卡一任务，最多 8 卡并行。
# 空闲判定同监控：util<5% 且 mem<3000MiB，且该卡未被本调度器占用。
# 队列清空且无在跑任务后，再等 30 分钟以防有新任务加入，随后退出。
set -u
ROOT=${MTP4TS_ROOT}
TSLIB=$ROOT/Time-Series-Library
Q=$ROOT/experiments/queue
RUN=$ROOT/experiments/running
DONE=$ROOT/experiments/done
LOGS=$ROOT/experiments/logs
N_GPU=8
EMPTY_LIMIT=30   # 队列连续空置的分钟数上限

free_gpus() {
  # 输出系统空闲且未被我们占用的 GPU 编号
  local sys_idle
  sys_idle=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{u=$2+0; m=$3+0; if (u<5 && m<3000) print $1+0}')
  for g in $sys_idle; do
    if ! grep -qs "^$g\$" "$RUN"/*.gpu 2>/dev/null; then
      echo "$g"
    fi
  done
}

sweep_done() {
  # 清理已结束的运行记录，打 done 标记
  for f in "$RUN"/*.run; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .run)
    pid=$(awk '{print $1}' "$f")
    if ! kill -0 "$pid" 2>/dev/null; then
      touch "$DONE/$name.done"
      rm -f "$f" "$RUN/$name.gpu"
      echo "[$(date '+%F %T')] job finished: $name"
    fi
  done
}

empty_min=0
echo "[$(date '+%F %T')] run_queue started, pid $$"
while true; do
  sweep_done
  launched=0
  for job in "$Q"/*.sh; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .sh)
    [ -e "$DONE/$name.done" ] && continue
    [ -e "$RUN/$name.run" ] && continue
    g=$(free_gpus | head -1)
    [ -z "${g:-}" ] && break
    echo "[$(date '+%F %T')] launch $name on GPU $g"
    (cd "$TSLIB" && CUDA_VISIBLE_DEVICES=$g nohup bash "$job" > "$LOGS/$name.log" 2>&1 &) 
    # 记录 pid：取刚启动的 bash 子进程
    sleep 3
    pid=$(pgrep -f "bash $job" | head -1)
    echo "${pid:-unknown}" > "$RUN/$name.run"
    echo "$g" > "$RUN/$name.gpu"
    launched=1
    sleep 5
  done
  n_run=$(ls "$RUN"/*.run 2>/dev/null | wc -l)
  n_pending=0
  for job in "$Q"/*.sh; do
    [ -e "$job" ] || continue
    name=$(basename "$job" .sh)
    [ -e "$DONE/$name.done" ] || [ -e "$RUN/$name.run" ] || n_pending=$((n_pending+1))
  done
  if [ "$n_pending" -eq 0 ] && [ "$n_run" -eq 0 ]; then
    empty_min=$((empty_min+1))
    if [ "$empty_min" -ge "$EMPTY_LIMIT" ]; then
      echo "[$(date '+%F %T')] queue empty for ${EMPTY_LIMIT}min, collecting results and exit"
      bash "$ROOT/experiments/collect_results.sh" >> "$LOGS/collect.log" 2>&1
      exit 0
    fi
  else
    empty_min=0
  fi
  sleep 60
done
