#!/usr/bin/env bash
set -euo pipefail
cd '/home/ubuntu/文档/ZHJ_UAV/ultralytics'
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
exec /home/ubuntu/miniconda3/envs/UAV_CV/bin/python -u -m experiments.c0_o2mc_5090.run pipeline --seed 0 --root '/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c0_o2mc_5090/5090-c0-o2mc-seeds012-20261104-r1/seed0' --output '/home/ubuntu/文档/ZHJ_UAV/ultralytics/docs/experiment_reports/2026-11-04_5090_c0_o2mc_completion/seed0'
