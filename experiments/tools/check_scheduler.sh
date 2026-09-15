#!/usr/bin/env bash
# 检查调度器是否存活并打印 pid（避免 pgrep 自匹配问题）
for p in $(pgrep -x python3); do
  if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'tools/scheduler.py'; then
    echo "scheduler alive: $p"
    exit 0
  fi
done
echo "scheduler dead"
exit 1
