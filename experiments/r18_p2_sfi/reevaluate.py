"""Re-evaluate the four original checkpoints without replacing the legacy evidence."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/r18_p2_sfi/corrected_20261024"


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    os.environ.update(PYTHONPATH=str(ROOT), MPLBACKEND="Agg", OMP_NUM_THREADS="4")
    status = {"status": "running", "completed": [], "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        for c in ("c0", "c2", "c1", "c3"):
            status["variant"] = c
            (OUT / "status.json").write_text(json.dumps(status, indent=2))
            run = ROOT / f"runs/detect/r18_p2_sfi_20261023/rtdetr-r18-{c}-150e-s0"
            with (ROOT / f"logs/p2-sfi-fixed-{c}-20261024.log").open("x") as log:
                subprocess.run([sys.executable, "-u", "experiments/r18_p2_sfi/evaluate.py", "--variant", c,
                                "--run", str(run), "--output-dir", str(OUT)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            status["completed"].append(c)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        (OUT / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
