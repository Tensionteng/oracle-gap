#!/usr/bin/env bash
# 延迟 6 小时启动器：等 GPU 空出后把 staged_x 的 job 放入队列并启动调度器。
# 用法：setsid nohup bash experiments/staged_x/launch_delayed.sh > experiments/logs/x_launcher.log 2>&1 &
set -u
ROOT=/mnt/jd/users/tengshiyuan.1/codes/mtp4ts
DELAY=${DELAY_SECONDS:-21600}   # 默认 6h
echo "[$(date '+%F %T')] launcher pid $$, sleeping ${DELAY}s"
sleep "$DELAY"
echo "[$(date '+%F %T')] waking up, staging x-series jobs"
cp "$ROOT"/experiments/staged_x/x*.sh "$ROOT"/experiments/queue/
if pgrep -f "run_queue.sh" > /dev/null; then
  echo "[$(date '+%F %T')] run_queue.sh already running, jobs will be picked up"
else
  echo "[$(date '+%F %T')] starting run_queue.sh"
  cd "$ROOT" && setsid nohup bash experiments/run_queue.sh >> experiments/logs/scheduler.log 2>&1 &
fi
echo "[$(date '+%F %T')] done"
