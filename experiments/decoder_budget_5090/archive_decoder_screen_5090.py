"""Archive CUDA timing samples and compact trained-depth evaluation evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def artifact(path):
    path = Path(path).resolve()
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--performance",nargs="+",required=True,type=Path)
    p.add_argument("--evaluation",nargs="+",required=True,type=Path)
    p.add_argument("--notes",required=True,type=Path,help="Compact tracked execution/warning record")
    p.add_argument("--validation",type=Path,help="Optional existing test/validation receipt")
    p.add_argument("--output",required=True,type=Path)
    args = p.parse_args()
    raw = args.output/"raw"
    raw.mkdir(parents=True,exist_ok=False)
    index = {"schema":"5090.decoder-screen-archive.v1","performance":[],"evaluation":[],
             "archive_policy":"Full raw CUDA samples retained. Raw wall samples stay in server assessment; per-block/pooled wall stats retained. All weights, inputs, predictions, raw stage tensors, COCO GT match lists and engines stay server-only."}
    for path in args.performance:
        source = json.loads(path.read_text())
        groups = source.pop("performance_groups")
        key = source["kind"]+"-"+source["variant"]
        identity = "identity-"+key+".json"
        source["server_assessment"] = artifact(path)
        source["archive_raw_wall_samples"] = False
        write(raw/identity,source)
        for workload in sorted({g["workload"] for g in groups.values()}):
            selected = {k:copy.deepcopy(g) for k,g in groups.items() if g["workload"]==workload}
            for g in selected.values():
                for backend in g["backends"].values():
                    for block in backend["blocks"]:
                        block.pop("wall_ms",None)
            filename = key+"-"+workload+".json"
            write(raw/filename,{"identity":identity,"groups":selected})
            index["performance"].append(filename)
    for path in args.evaluation:
        source = json.loads(path.read_text())
        key = "depth-eval-"+source["variant"]
        filename = key+".json"
        source["server_summary"] = artifact(path)
        write(raw/filename,source)
        index["evaluation"].append(filename)
        for original in ("formal_gt_transitions.csv","diagnostic_transitions.csv","diagnostic_summary.json"):
            shutil.copyfile(path.parent/original,raw/(source["variant"]+"-"+original))
        pin = (path.parent/"source_pin.json").read_bytes()
        pin_target = raw/"source_pin.json"
        if pin_target.exists() and pin_target.read_bytes()!=pin:
            raise ValueError("Models used different source pins; archive explicitly rather than overwrite")
        pin_target.write_bytes(pin)
    write(raw/"index.json",index)
    write(raw/"execution_notes.json",json.loads(args.notes.read_text()))
    if args.validation:
        write(raw/"validation.json",json.loads(args.validation.read_text()))
    print("Archive bytes",sum(p.stat().st_size for p in raw.iterdir()),"files",len(list(raw.iterdir())))


if __name__ == "__main__":
    main()
