"""Cross-seed adaptation of screen_decoder_eval_5090.py; original preserved.

Trained D1/D2/D3 early-exit AP and same-GT transition evaluation; no training.

Uses native DETRScope observation for all executed layers, then proves on every
image that genuine native early exits reproduce those intermediate predictions.
Large predictions/records/COCO GT matches remain on the server.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import torch

from experiments.decoder_budget_5090.profile_runtime_5090 import ROOT, DEFAULT_DATA, DEFAULT_RUN, configure, environment, sha, source_identity
from experiments.decoder_budget_5090.screen_decoder_perf_5090 import checkpoint_path, load_model, source_hashes
from experiments.decoder_budget_5090.run import batches, input_identity, write


def csv_write(path, rows):
    with path.open("w",newline="") as stream:
        w = csv.DictWriter(stream,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def artifact(path):
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}


def pooled_formal_changes(before, after, annotations, label):
    from detrscope.evaluation.coco import compare_gt
    comparison = compare_gt(before,after)
    counters = defaultdict(lambda:defaultdict(int))
    for row in comparison["thresholds"]:
        threshold = round(row["iou"],2)
        for state,ids in row["gt_ids"].items():
            for identity in ids:
                a = annotations[str(identity)]
                area = a["area"]
                size = "small" if area < 1024 else "medium" if area < 9216 else "large"
                for group in ("all",f"category:{a['category_id']}","size:"+size):
                    counters[(threshold,group)][state] += 1
    rows = []
    for (threshold,group),counts in sorted(counters.items()):
        counts = {name:counts[name] for name in ("lost","gained","stable_tp","stable_fn")}
        denominator = sum(counts.values())
        rows.append({"transition":label,"iou":threshold,"group":group,"gt_denominator":denominator,
                     **counts,"before_tp":counts["lost"]+counts["stable_tp"],
                     "after_tp":counts["gained"]+counts["stable_tp"]})
    return comparison, rows


def trajectory_diagnostics(raw, output):
    """Reuse SDK matching/trajectory definitions while retaining compact pooled counters."""
    from detrscope import AnalysisConfig
    from detrscope.core.records import iter_batches
    from detrscope.analysis.trajectory import build_trajectories
    from detrscope.analysis.degradation import analyze_image
    from detrscope.analysis.metrics import aggregate
    config = AnalysisConfig()
    counts = defaultdict(lambda:[0,0])
    images = gt_count = fixed_pairs = excluded = events = 0
    for payload in iter_batches(raw):
        for image in payload["images"]:
            trajectories = build_trajectories(image,config,payload["model"])
            changes = analyze_image(trajectories,config)
            for row in aggregate(changes["events"],["D1->D2","D2->D3"],config):
                key = tuple(row[k] for k in ("family","transition","class_id","object_size","event","denominator_unit"))
                counts[key][0] += row["numerator"]
                counts[key][1] += row["denominator"]
            images += 1; gt_count += len(image["targets"]["gt_ids"])
            fixed_pairs += len(trajectories["fixed"])
            excluded += len(changes["excluded_transitions"])
            events += len(changes["events"])
            if images % 50 == 0:
                print("trajectory images",images,flush=True)
    rows = [{**dict(zip(("family","transition","class_id","object_size","event","denominator_unit"),key)),
             "numerator":value[0],"denominator":value[1],"rate":value[0]/value[1] if value[1] else None}
            for key,value in sorted(counts.items(),key=lambda p:tuple(map(str,p[0])))]
    csv_write(output/"diagnostic_transitions.csv",rows)
    result = {"images":images,"gt_count":gt_count,"fixed_pair_count":fixed_pairs,"event_count":events,
              "excluded_transitions":excluded,"config":config.to_dict(),
              "semantics":"SDK Hungarian iou*(1-IoU)+score*(1-pGT), no confidence/IoU gate. Fixed query-GT pairs use assignment union plus D1 fallback. GT trajectories reassign each layer. Neither is COCO TP/FP/AP."}
    write(output/"diagnostic_summary.json",result)
    return result


def run(args):
    import cv2
    from detrscope import Debugger
    from detrscope.adapters.sodetr_data import CocoTargets
    from detrscope.core.io import state_sha256
    from detrscope.core.records import iter_batches
    from detrscope.evaluation.coco import CocoEvaluator
    from ultralytics.models.rtdetr.predict import RTDETRPredictor
    from experiments.decoder_budget_5090.source_adapter import create_source_pin, PinnedRTDETRAdapter
    configure()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    annotations_path = DEFAULT_DATA/"annotations/VisDrone2019-DET_val_coco.json"
    content = json.loads(annotations_path.read_text())
    checkpoint = args.checkpoint.resolve()
    baseline = json.loads(args.baseline_evaluation.read_text())
    if artifact(checkpoint)["sha256"] != baseline["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint differs from completed training/evaluation receipt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = (saved.get("ema") or saved["model"]).float().eval().cuda()
    head = model.model[-1]
    if head.decoder.num_layers != 3 or head.decoder.eval_idx != 2 or head.num_queries != 300:
        raise RuntimeError("Expected the trained 3-layer/300Q checkpoint")
    names = [str(model.names[i]) for i in range(len(model.names))]
    coco = CocoTargets(annotations_path,names)
    category_ids = [next(cid for cid,index in coco.categories.items() if index==i) for i in range(len(names))]
    images = sorted(content["images"],key=lambda r:r["id"])
    if args.limit:
        images = images[:args.limit]
    paths = [DEFAULT_DATA/"images/val"/Path(im["file_name"]).name for im in images]
    pin_path = output/"source_pin.json"
    create_source_pin(ROOT,pin_path,expected_commit=args.expected_commit)
    adapter = PinnedRTDETRAdapter(model,repo=ROOT,pin=pin_path,class_names=names,category_ids=category_ids)
    before = state_sha256(model)
    summary = {"schema":"5090.decoder-depth-evaluation.v1","variant":args.variant,"training_seed":args.seed,
               "started_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "source":source_identity(ROOT),"source_sha256":source_hashes((Path(__file__),)),
               "environment":environment("cuda:0"),"checkpoint":artifact(checkpoint),
               "protocol":{"images":len(paths),"full_validation":not args.limit,"batch":args.batch,"size":640,
                           "dtype":"float32","fused":False,"autocast":False,"queries":300,"points_per_level":[4,4,4],
                           "preprocess":"native LetterBox scale_fill 640; RGB BCHW /255",
                           "postprocess":"native RTDETRPredictor; per-query argmax class; conf>0.001; score-sort; max_det=300; no rounding; no NMS",
                           "official_eval":"pycocotools.COCOeval bbox maxDets=[1,10,100], IoU=.50:.05:.95",
                           "annotation":artifact(annotations_path),"category_ids":category_ids,"class_names":names,
                           "seed":20261024,"model_state_sha256":before,"source_pin":artifact(pin_path),
                           "scope":"D1/D2 auxiliary trained readouts; native eval_idx truly exits after each specified layer; no retraining"}}
    write(output/"progress.json",summary)
    debug = Debugger(model,adapter,output/"observed-d3")
    manifest = debug.observe(batches(paths,coco,"cuda:0",args.batch,640),probes=[],provenance=summary["protocol"])
    if not all(b["identity_passed"] and b["restoration_passed"] for b in manifest["batches"]):
        raise RuntimeError("Observation equivalence failed")
    summary["observation"] = {"images":manifest["image_count"],"batches":len(manifest["batches"]),
                              "all_identity_and_restore_passed":True,"manifest":artifact(debug.output_dir/"manifest.json"),
                              "input_gt_identity_sha256":input_identity(debug.output_dir)}
    print(args.variant,"observed images",manifest["image_count"],flush=True)
    predictor = RTDETRPredictor(overrides={"imgsz":640,"conf":.001,"max_det":300,"verbose":False})
    predictor.imgsz = [640, 640]
    predictor.model = SimpleNamespace(names=model.names)
    predictions = {d:[] for d in (1,2,3)}
    max_differences = {d:0. for d in (1,2,3)}
    execution_counts = {d:set() for d in (1,2,3)}
    image_ids = []
    for index,payload in enumerate(iter_batches(debug.output_dir)):
        records = payload["images"]
        part = [Path(im["metadata"]["image_path"]) for im in records]
        batch = next(batches(part,coco,"cuda:0",args.batch,640))
        input_sha = hashlib.sha256(batch.inputs.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if input_sha != manifest["batches"][index]["input_sha256"]:
            raise RuntimeError("Native early-exit replay input differs from observed forward")
        originals = [cv2.imread(str(p)) for p in part]
        predictor.batch = ([str(p) for p in part],originals,[""]*len(part))
        for depth in (1,2,3):
            head.decoder.eval_idx = depth-1
            seen,handles = [],[]
            for i,layer in enumerate(head.decoder.layers):
                handles.append(layer.register_forward_hook(lambda m,a,y,i=i:seen.append(i+1)))
            try:
                native = model(batch.inputs)[0]
            finally:
                for h in handles:
                    h.remove()
                head.decoder.eval_idx = 2
            if seen != list(range(1,depth+1)):
                raise RuntimeError("A later Decoder layer executed despite early exit")
            execution_counts[depth].add(tuple(seen))
            stages = [next(s for s in im["stages"] if s["source"]=="decoder" and s["layer_id"]==depth) for im in records]
            observed = torch.stack([torch.cat((s["boxes"],s["probabilities"]),-1) for s in stages]).cuda()
            difference = (native-observed).abs().max().item()
            max_differences[depth] = max(max_differences[depth],difference)
            torch.testing.assert_close(native,observed,rtol=0,atol=0)
            results = predictor.postprocess((native,None),None,originals)
            for image,result in zip(records,results):
                for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                    x1,y1,x2,y2 = box
                    predictions[depth].append({"image_id":image["metadata"]["image_id"],"category_id":category_ids[int(cls)],
                                               "bbox":[x1,y1,x2-x1,y2-y1],"score":score})
        image_ids.extend(im["metadata"]["image_id"] for im in records)
        if (index+1)%25==0:
            print(args.variant,"early-exit verified images",len(image_ids),flush=True)
    summary["early_exit_equivalence"] = {"images":len(image_ids),"all_exact":True,
                                         "max_abs":max_differences,"executed_layers":{d:[list(v) for v in sorted(a)] for d,a in execution_counts.items()}}
    evaluator = CocoEvaluator(annotations_path)
    formal,conditions = {},{}
    for depth in (1,2,3):
        path = output/f"d{depth}-predictions.json"
        write(path,predictions[depth])
        formal[depth] = evaluator.evaluate(predictions[depth],image_ids=image_ids)
        coco_path = output/f"d{depth}-coco.json"
        write(coco_path,formal[depth])
        conditions[f"d{depth}"] = {"depth":depth,"metrics":formal[depth]["metrics"],"predictions":artifact(path),
                                   "coco":artifact(coco_path),"gt_denominator":len(formal[depth]["gt_states"]),
                                   "prediction_count":len(predictions[depth])}
        print(args.variant,"D",depth,formal[depth]["metrics"],flush=True)
    if not args.limit:
        metric_diffs = {k: abs(formal[3]["metrics"][k] - baseline["metrics"][k]) for k in formal[3]["metrics"]}
        if max(metric_diffs.values()) > 1e-12:
            raise RuntimeError(f"D3 differs from historical same-checkpoint evaluation: {metric_diffs}")
        summary["historical_d3_replay"] = {"all_metrics_within_1e12": True, "metric_max_abs": max(metric_diffs.values()),
            "previous": artifact(args.baseline_evaluation)}
    annotations = {str(a["id"]):a for a in content["annotations"]}
    rows = []
    for a,b in ((1,2),(2,3),(1,3)):
        comparison,pooled = pooled_formal_changes(formal[a],formal[b],annotations,f"D{a}->D{b}")
        path = output/f"d{a}-d{b}-formal-gt-changes.json"
        write(path,comparison)
        rows.extend(pooled)
    csv_write(output/"formal_gt_transitions.csv",rows)
    summary["conditions"] = conditions
    summary["formal_protocol"] = formal[3]["protocol"]
    write(output/"progress.json",summary)
    summary["diagnostics"] = trajectory_diagnostics(debug.output_dir,output)
    if state_sha256(model) != before or head.decoder.eval_idx != 2 or head.num_queries != 300:
        raise RuntimeError("Evaluation changed trained model state/budget")
    hooks = sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())
    if hooks:
        raise RuntimeError("Hooks leaked")
    summary.update(model_state_unchanged=True,hooks_remaining=0,eval_idx_restored=2,
                   finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),status="completed")
    write(output/"summary.json",summary)
    print("OUTPUT",output,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",required=True,choices=["c0","c2"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-evaluation", type=Path, required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--batch",type=int,default=2)
    args = p.parse_args()
    if args.limit<0 or args.batch!=2:
        p.error("Require limit>=0 and established batch=2")
    with torch.inference_mode():
        run(args)


if __name__ == "__main__":
    main()
