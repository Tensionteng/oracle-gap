#!/bin/bash
# Run a command-list file with at most P concurrent jobs (GPU id is baked
# into each line via CUDA_VISIBLE_DEVICES, assigned round-robin by gen_cmds.py).
# usage: bash run_cmds.sh <cmds.txt> [P]
set -u
CMDS=$1
P=${2:-24}
xargs -P "$P" -I{} bash -c "{}" < "$CMDS"
echo "DONE $CMDS"
