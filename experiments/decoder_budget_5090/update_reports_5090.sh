#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
UAV_REPORT_PYTHON="${UAV_REPORT_PYTHON:-/home/ubuntu/miniconda3/envs/UAV_CV/bin/python}"
exec "$UAV_REPORT_PYTHON" experiments/decoder_budget_5090/report_5090.py \
  --runs-dir "${1:-runs/decoder_budget_5090}" --output "${2:-reports/5090}"
