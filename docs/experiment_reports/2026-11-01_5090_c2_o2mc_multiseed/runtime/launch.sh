#!/usr/bin/env bash
set -uo pipefail
cd '/home/ubuntu/文档/ZHJ_UAV/ultralytics'
export PYTHONPATH="$PWD:$PWD/../detrscope/src"
/home/ubuntu/miniconda3/envs/UAV_CV/bin/python -u -m experiments.c2_o2mc_5090.multiseed_queue --root '/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1' --output '/home/ubuntu/文档/ZHJ_UAV/ultralytics/docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed'
code=$?
printf '{"returncode":%s}\n' "$code" > '/home/ubuntu/文档/ZHJ_UAV/ultralytics/runs/c2_o2mc_5090/5090-c2-o2mc-multiseed-20261101-r1/exit.json'
exit "$code"
