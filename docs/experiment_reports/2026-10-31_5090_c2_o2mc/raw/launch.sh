#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/文档/ZHJ_UAV/ultralytics
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
export YOLO_CONFIG_DIR="$PWD/runs/c2_o2mc_5090/ultralytics-config"
export PYTHONUNBUFFERED=1
set +e
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -u -m experiments.c2_o2mc_5090.run pipeline --root "$PWD/runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2"
run_exit=$?
printf '{"exit_code":%s,"finished":"%s"}\n' "$run_exit" "$(date -Is)" > runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2/exit.json
exit "$run_exit"
