#!/usr/bin/env bash
# GPU 空闲监控：全部 8 张卡连续空闲 10 分钟则触发实验队列。
# 空闲判定：util < 5% 且 memory.used < 3000 MiB。
# 用法: nohup bash tools/gpu_monitor.sh > logs/gpu_monitor.log 2>&1 &
set -u
ROOT=${MTP4TS_ROOT}
STATE_DIR=$ROOT/experiments
PIDFILE=$STATE_DIR/gpu_monitor.pid
INTERVAL=60          # 每分钟检查一次
NEED_IDLE_MIN=10     # 需要连续空闲的分钟数
N_GPU=8

if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "[$(date '+%F %T')] monitor already running (pid $(cat $PIDFILE)), exit"
  exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f $PIDFILE' EXIT

idle_streak=0
echo "[$(date '+%F %T')] monitor started, pid $$, waiting for $N_GPU GPUs to be idle for ${NEED_IDLE_MIN}min"

while true; do
  stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$stats" ]; then
    echo "[$(date '+%F %T')] WARN: nvidia-smi failed, keep waiting"
    idle_streak=0
    sleep $INTERVAL; continue
  fi
  n_idle=$(echo "$stats" | awk -F',' '{u=$1+0; m=$2+0; if (u<5 && m<3000) c++} END{print c+0}')
  if [ "$n_idle" -ge "$N_GPU" ]; then
    idle_streak=$((idle_streak+1))
    echo "[$(date '+%F %T')] all $N_GPU GPUs idle, streak ${idle_streak}/${NEED_IDLE_MIN} min"
  else
    if [ "$idle_streak" -gt 0 ]; then
      echo "[$(date '+%F %T')] idle broken ($n_idle/$N_GPU idle), reset streak"
    fi
    idle_streak=0
  fi
  if [ "$idle_streak" -ge "$NEED_IDLE_MIN" ]; then
    echo "[$(date '+%F %T')] idle threshold reached, launching run_queue.sh"
    nohup bash "$ROOT/experiments/run_queue.sh" > "$ROOT/experiments/logs/run_queue.log" 2>&1 &
    echo "[$(date '+%F %T')] queue launched (pid $!), monitor exits"
    exit 0
  fi
  sleep $INTERVAL
done
