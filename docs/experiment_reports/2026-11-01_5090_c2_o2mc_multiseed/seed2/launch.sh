#!/usr/bin/env bash
set -euo pipefail
cd '/home/ubuntu/文档/ZHJ_UAV/ultralytics'
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
exec /home/ubuntu/miniconda3/envs/UAV_CV/bin/python -u -m experiments.c2_o2mc_5090.multiseed_run pipeline --seed 2 --root '/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1/seed2' --output '/home/ubuntu/文档/ZHJ_UAV/ultralytics/docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed/seed2'
