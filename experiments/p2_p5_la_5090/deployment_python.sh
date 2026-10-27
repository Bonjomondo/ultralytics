#!/usr/bin/env bash
# A regular executable path survives the historical controller's Path.resolve().
set -euo pipefail
: "${UAV_TRT_PYTHON:?Set UAV_TRT_PYTHON to the TensorRT virtualenv Python entry point}"
exec "$UAV_TRT_PYTHON" "$@"
