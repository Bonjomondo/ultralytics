"""Same-session FP32 forward timings for B0, C2 and the scalar candidate."""
from pathlib import Path
import argparse,json,hashlib,random
import numpy as np
import torch
from ultralytics import RTDETR

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    b=ROOT.parent
    weights={'B0':b/'rtdetr-r18-sodetr-paper-20261027/runs/rtdetr-r18-author-recipe-350e-s0/weights/best.pt',
             'C2':b/'rtdetr-r18-c2-author-recipe-20261028/runs/rtdetr-r18-c2-matched-b0-350e-s0/weights/best.pt',
             'scalar':a.run/'weights/best.pt'}
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    models={};out={};x=torch.rand(1,3,640,640,device='cuda')
    for label,path in weights.items():
        m=RTDETR(str(path)).model.float().cuda().eval();out[label]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parameters':sum(p.numel() for p in m.parameters()),'samples_ms':[]};models[label]=m.fuse().eval()
    rng=random.Random(4)
    with torch.inference_mode():
        for m in models.values():
            for _ in range(2 if a.smoke else 50):m(x)
        for _ in range(1 if a.smoke else 5):
            order=list(models);rng.shuffle(order)
            for label in order:
                for _ in range(8 if a.smoke else 100):
                    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);s.record();models[label](x);e.record();e.synchronize();out[label]['samples_ms'].append(float(s.elapsed_time(e)))
    for d in out.values():d.update(mean_ms=float(np.mean(d['samples_ms'])),median_ms=float(np.median(d['samples_ms'])),p95_ms=float(np.percentile(d['samples_ms'],95)))
    (a.run/'evaluation/benchmark.json').write_text(json.dumps({'models':out,'smoke':a.smoke,'protocol':'640 batch1 fused FP32 GPU forward, same session randomized order, excludes transfer/pre/postprocess'},indent=2)+'\n')
if __name__=='__main__':main()
