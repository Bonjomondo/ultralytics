"""Create a compact, readable archive; full traces/exports remain on the server."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", nargs="+", required=True, help="variant=server assessment.json, one independent process per primary variant")
    p.add_argument("--deployment", nargs="*", default=[], type=Path)
    p.add_argument("--notes", type=Path, help="Tracked compact warning/history record to carry into a rebuilt archive")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    provenance = {"schema": "5090.runtime-archive.v1", "runtime": {}, "deployment": [],
                  "boundary": "Only files inside this archive and linked source scripts are cloud artifacts. Absolute server paths are references, not required inputs for offline analysis."}
    for specification in args.runtime:
        variant, filename = specification.split("=", 1)
        path = Path(filename).resolve()
        source = json.loads(path.read_text())
        model = copy.deepcopy(source["models"][variant])
        identity = {k:v for k,v in source.items() if k != "models"}
        identity["server_assessment"] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        provenance["runtime"][variant] = identity
        timeline, cpu_rows = [], []
        for condition, data in model["conditions"].items():
            for mode in ("plain", "modules"):
                key = "profile_"+mode
                if key not in data:
                    continue
                profile = data.pop(key)
                for r in profile.pop("first_iteration_timeline"):
                    timeline.append({"condition": condition, "mode": mode,
                                     **{k:v for k,v in r.items() if k != "kernel"}})
                trace = json.loads(Path(profile["trace"]["server_path"]).read_text())["traceEvents"]
                iteration_ranges = [e for e in trace if e.get("name", "").startswith("iteration::") and e.get("ph") == "X"]
                buckets = {}
                for e in trace:
                    if e.get("cat") not in ("cpu_op", "user_annotation") or e.get("ph") != "X":
                        continue
                    if not any(i["tid"] == e["tid"] and i["ts"] <= e["ts"] and e["ts"]+e["dur"] <= i["ts"]+i["dur"]+.01 for i in iteration_ranges):
                        continue
                    b = buckets.setdefault((e["cat"],e["name"]), [0,0.])
                    b[0] += 1
                    b[1] += e["dur"]
                for (category,name),(calls,duration) in buckets.items():
                    cpu_rows.append({"condition": condition, "mode": mode, "category": category, "name": name,
                                     "calls_per_forward": calls/len(iteration_ranges),
                                     "inclusive_cpu_us_per_forward": duration/len(iteration_ranges)})
                filename = f"profile-{variant}-{condition}-{mode}.json"
                write(raw/filename, profile)
                data[key] = filename
        write(raw/f"runtime-{variant}.json", model)
        csv_write(raw/f"timeline-{variant}.csv", timeline)
        csv_write(raw/f"cpu-{variant}.csv", cpu_rows)
    for path in args.deployment:
        source = json.loads(path.read_text())
        for variant, model in source["models"].items():
            timeline = []
            for condition, data in model["conditions"].items():
                if "profile_plain" not in data:
                    continue
                profile = data.pop("profile_plain")
                for r in profile.pop("first_iteration_timeline"):
                    timeline.append({"condition": condition, "mode": "plain",
                                     **{k:v for k,v in r.items() if k != "kernel"}})
                filename = f"profile-deployment-{variant}-{condition}.json"
                write(raw/filename, profile)
                data["profile_plain"] = filename
            write(raw/f"deployment-{variant}.json", model)
            csv_write(raw/f"deployment-timeline-{variant}.csv", timeline)
        provenance["deployment"].append({**{k:v for k,v in source.items() if k != "models"},
                                          "server_assessment": {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}})
    write(raw/"identity.json", provenance)
    if args.notes:
        write(raw/"backend_notes.json", json.loads(args.notes.read_text()))
    print("Archive",args.output,"bytes",sum(p.stat().st_size for p in raw.iterdir()))


if __name__ == "__main__":
    main()
