"""Check completed LA-only and four-arm evidence using tracked text only."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.p2_p5_la_5090.archive_la_only_results import validate,verify_manifest

if __name__=='__main__':
    archive=Path(__file__).resolve().parent
    validate(archive)
    verify_manifest(archive)
