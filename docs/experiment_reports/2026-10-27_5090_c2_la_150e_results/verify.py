"""Verify final result with tracked text only; no torch/GPU/server data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.p2_p5_la_5090.archive_joint_results import validate, verify_manifest

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    validate(here)
    verify_manifest(here)
