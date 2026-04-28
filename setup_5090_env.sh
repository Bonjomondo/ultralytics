#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then
    exec bash "$0" "$@"
  fi
  printf '%s\n' '[ERROR] This script requires bash, but bash was not found in PATH.' >&2
  exit 1
fi

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-ultralytics-5090}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SELF_CHECK_MODEL="${SELF_CHECK_MODEL:-${REPO_ROOT}/yolo11s.pt}"
SELF_CHECK_IMAGE="${SELF_CHECK_IMAGE:-${REPO_ROOT}/ultralytics/assets/bus.jpg}"

TORCH_CANDIDATES=(
  "2.8.0|0.23.0|https://download.pytorch.org/whl/cu128"
  "2.7.0|0.22.0|https://download.pytorch.org/whl/cu128"
  "2.6.0|0.21.0|https://download.pytorch.org/whl/cu124"
)

log() {
  printf '%s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "[ERROR] Missing required command: $1"
    exit 1
  fi
}

ensure_conda_shell() {
  require_command conda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

env_exists() {
  conda env list | awk 'NR > 1 {print $1}' | grep -qx "${ENV_NAME}"
}

create_env_if_needed() {
  if env_exists; then
    log "[INFO] Reusing existing conda environment: ${ENV_NAME}"
    return
  fi

  log "[INFO] Creating conda environment: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
}

activate_env() {
  conda activate "${ENV_NAME}"
}

install_torch_stack() {
  local candidate torch_version torchvision_version index_url
  local install_ok=0

  for candidate in "${TORCH_CANDIDATES[@]}"; do
    IFS='|' read -r torch_version torchvision_version index_url <<<"${candidate}"
    log "[INFO] Trying torch=${torch_version}, torchvision=${torchvision_version} from ${index_url}"

    if python -m pip install --upgrade --index-url "${index_url}" "torch==${torch_version}" "torchvision==${torchvision_version}"; then
      if python - <<'PY'
import torch
import sys

if not torch.cuda.is_available():
    sys.exit(2)

props = torch.cuda.get_device_properties(0)
arch = f"sm_{props.major}{props.minor}"
if arch not in set(torch.cuda.get_arch_list()):
    sys.exit(3)
PY
      then
        log "[INFO] Selected torch=${torch_version}, torchvision=${torchvision_version}"
        install_ok=1
        break
      fi
      log "[WARN] Installed torch stack does not support the current GPU architecture; trying next candidate."
    else
      log "[WARN] Candidate install failed; trying next candidate."
    fi
  done

  if [[ "${install_ok}" != "1" ]]; then
    log "[ERROR] Could not install a torch/torchvision pair that supports the current GPU."
    log "[ERROR] Set TORCH_CANDIDATES manually in this script if your network mirror requires a different wheel set."
    exit 1
  fi
}

install_editable_ultralytics() {
  log "[INFO] Installing ultralytics in editable mode from ${REPO_ROOT}"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${REPO_ROOT}"
  python -m pip check
}

run_self_check() {
  if [[ "${RUN_SMOKE}" != "1" ]]; then
    log "[INFO] RUN_SMOKE=0, skipping runtime self-check"
    return
  fi

  if [[ ! -f "${SELF_CHECK_MODEL}" ]]; then
    log "[ERROR] Missing self-check model: ${SELF_CHECK_MODEL}"
    exit 1
  fi

  if [[ ! -f "${SELF_CHECK_IMAGE}" ]]; then
    log "[ERROR] Missing self-check image: ${SELF_CHECK_IMAGE}"
    exit 1
  fi

  REPO_ROOT="${REPO_ROOT}" SELF_CHECK_MODEL="${SELF_CHECK_MODEL}" SELF_CHECK_IMAGE="${SELF_CHECK_IMAGE}" python - <<'PY'
import os
import sys
from pathlib import Path

import torch

repo_root = Path(os.environ["REPO_ROOT"])
model_path = Path(os.environ["SELF_CHECK_MODEL"])
image_path = Path(os.environ["SELF_CHECK_IMAGE"])

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment")

props = torch.cuda.get_device_properties(0)
arch = f"sm_{props.major}{props.minor}"
arch_list = torch.cuda.get_arch_list()

print(f"gpu={props.name}")
print(f"capability={props.major}.{props.minor}")
print(f"arch_list={arch_list}")

if arch not in arch_list:
    raise SystemExit(f"Current torch build does not support {arch}")

from ultralytics import YOLO

model = YOLO(str(model_path))
results = model.predict(source=str(image_path), imgsz=320, device=0, verbose=False)

print(f"self_check_ok={len(results)}")
print(f"repo_root={repo_root}")
PY
}

main() {
  ensure_conda_shell
  create_env_if_needed
  activate_env
  install_torch_stack
  install_editable_ultralytics
  run_self_check

  log "[INFO] Environment is ready. Activate it with: conda activate ${ENV_NAME}"
  log "[INFO] To start the batch run: bash run_5090_experiments.sh"
}

main "$@"