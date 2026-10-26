"""Isolated stdlib verification; no torch/GPU/SDK/server assets required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.decoder_budget_5090.report_fresh_results_5090 import validate, verify_manifest

if __name__ == '__main__':
    archive = Path(__file__).resolve().parent
    validate(archive)
    verify_manifest(archive)
