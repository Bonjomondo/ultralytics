"""Evaluate only this task's newly trained checkpoint, with native FP32 COCO AP."""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from experiments.decoder_budget_5090.profile_runtime_5090 import (
    DEFAULT_DATA, ROOT, configure, environment, load_input, measure, stats,
)
from experiments.decoder_budget_5090.run import batches, write
from experiments.decoder_budget_5090.screen_decoder_perf_5090 import PredictionOnly, ready_paths
from experiments.decoder_budget_5090.train_fresh_budgets_5090 import BUDGETS,inspect_architecture
from experiments.decoder_budget_5090.train_5090 import sha256_file


def run(args):
    import cv2
    from detrscope.adapters.sodetr_data import CocoTargets
    from detrscope.core.io import state_sha256
    from detrscope.evaluation.coco import CocoEvaluator
    from ultralytics.models.rtdetr.predict import RTDETRPredictor
    configure()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=(saved.get("ema") or saved["model"]).float().eval().cuda()
    architecture=inspect_architecture(model,args.variant)
    initial=state_sha256(model)
    ann=DEFAULT_DATA/"annotations/VisDrone2019-DET_val_coco.json"
    content=json.loads(ann.read_text())
    names=[str(model.names[i]) for i in range(len(model.names))]
    coco=CocoTargets(ann,names)
    categories=[next(k for k,v in coco.categories.items() if v==i) for i in range(len(names))]
    images=sorted(content["images"],key=lambda x:x["id"])
    if args.limit:images=images[:args.limit]
    paths=[DEFAULT_DATA/"images/val"/Path(im["file_name"]).name for im in images]
    predictor=RTDETRPredictor(overrides={"imgsz":640,"conf":.001,"max_det":300,"verbose":False})
    predictor.model=SimpleNamespace(names=model.names)
    record={"schema":"5090.fresh-budget-evaluation.v1","variant":args.variant,"scope":"subset_gate" if args.limit else "full_val",
            "status":"running","architecture":architecture,"environment":environment("cuda:0"),
            "checkpoint":{"path":str(args.checkpoint.resolve()),"bytes":args.checkpoint.stat().st_size,"sha256":sha256_file(args.checkpoint)},
            "protocol":{"dtype":"float32","fused":False,"batch":2,"size":640,"conf":.001,"max_det":300,
                        "formal_max_dets":[1,10,100],"annotation_path":str(ann),"annotation_sha256":sha256_file(ann),
                        "images":len(images),"image_ids":[im["id"] for im in images],"category_ids":categories,
                        "preprocess":"native scale_fill RGB BCHW /255","postprocess":"native per-query argmax, sorted confidence; no JSON rounding/no NMS"}}
    write(out/"progress.json",record)
    predictions=[];ids=[]
    first=True
    for batch in batches(paths,coco,"cuda:0",2,640):
        output=model(batch.inputs)[0]
        if first:
            torch.testing.assert_close(output,model(batch.inputs)[0],atol=0,rtol=0)
            record["repeated_forward_exact"]=True;first=False
        originals=[cv2.imread(m["image_path"]) for m in batch.metadata]
        predictor.batch=([m["image_path"] for m in batch.metadata],originals,[""]*len(originals))
        results=predictor.postprocess((output,None),None,originals)
        for m,r in zip(batch.metadata,results):
            ids.append(m["image_id"])
            for box,score,cls in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.cpu().tolist()):
                x1,y1,x2,y2=box
                predictions.append({"image_id":m["image_id"],"category_id":categories[int(cls)],
                                    "bbox":[x1,y1,x2-x1,y2-y1],"score":score})
    write(out/"predictions.json",predictions)
    formal=CocoEvaluator(ann).evaluate(predictions,image_ids=ids)
    write(out/"coco.json",formal)
    record.update(metrics=formal["metrics"],gt_denominator=len(formal["gt_states"]),
                  predictions_sha256=sha256_file(out/"predictions.json"),coco_sha256=sha256_file(out/"coco.json"))
    if state_sha256(model)!=initial:
        raise RuntimeError("Native evaluation changed checkpoint state")
    record["model_state_unchanged"]=True
    if args.benchmark:
        x,input_identity=load_input(DEFAULT_DATA)
        fn,backends,keep=ready_paths(PredictionOnly(model),[x],["images"],out/"deployment",True,
                                   ["eager_fp32","trt_fp32","trt_fp16"])
        for f in fn.values():
            for _ in range(150):f()
        torch.cuda.synchronize()
        for r in range(10):
            order=list(fn);random.Random(20261025+r).shuffle(order)
            for key in order:
                block=measure(fn[key],10,50);block["round"]=r+1;backends[key]["blocks"].append(block)
        for key in fn:
            b=backends[key]
            b["cuda_stats_ms"]=stats([v for a in b["blocks"] for v in a["cuda_ms"]])
            b["wall_stats_ms"]=stats([v for a in b["blocks"] for v in a["wall_ms"]])
            b["status"]="completed"
        record["benchmark"]={"status":"completed" if len(fn)==3 else "partial_failed","input":input_identity,
                             "scope":"resident GPU batch1 model forward; no pre/postprocess or transfers; FP16 AP not evaluated",
                             "rounds":10,"samples_per_backend":500,"backends":backends}
    record["status"]="completed"
    write(out/"summary.json",record)
    print(args.variant,record["scope"],record["metrics"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",choices=BUDGETS,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--benchmark",action="store_true")
    a=p.parse_args()
    if a.limit<0:p.error("limit must be nonnegative")
    with torch.inference_mode():
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):run(a)
        stream.synchronize()
