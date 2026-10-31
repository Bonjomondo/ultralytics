"""Serial inference-only diagnostics; GPU lock, immutable assets, no training."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
STABILITY = ROOT/'runs/p2_stability_5090/5090-p2-stability-20261027-r1'
HERE = Path(__file__).parent
ANCESTOR = '6d1d4d1badbecf47970243685a1d712ad3e31559'

def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    q=p.with_suffix('.tmp'); q.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); q.replace(p)
def now(): return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    state={'status':'preflight','started_utc':now(),'stages':[],'no_training':True}
    write(out/'status.json',state)
    try:
        import torch, ultralytics, detrscope
        if not torch.cuda.is_available() or '5090' not in torch.cuda.get_device_name(0): raise RuntimeError('Require 5090')
        assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics')
        assert Path(detrscope.__file__).resolve().is_relative_to(ROOT.parent/'detrscope/src')
        state['environment']={'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':str(torch.__version__),'ultralytics':ultralytics.__file__,'detrscope':detrscope.__file__,'gpu':torch.cuda.get_device_name(0)}
        pin=read(STABILITY/'plan.json')
        assert all(sha(ROOT/k)==v for k,v in pin['source_sha256'].items()), 'Frozen source changed'
        annotations=ROOT.parent/'SO_DETR/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json'
        assert sha(annotations)==pin['annotations_sha256']
        state['annotation_sha256']=sha(annotations)
        assets=[]
        for seed in (1,2):
            for arm in ('c0','c2'):
                t=STABILITY/f'training/5090-{arm}-150e-s{seed}'
                e=STABILITY/f'evaluations/{arm}-s{seed}/summary.json'
                train,ev=read(t/'experiment.json'),read(e)
                assert train['status']==ev['status']=='completed' and train['completed_epochs']==150
                assert train['seed']==ev['seed']==seed and train['variant']==ev['variant']==arm
                assert train['checkpoints']['best']==ev['checkpoint']
                assert sha(t/'weights/best.pt')==ev['checkpoint']['sha256']
                assert sha(t/'experiment.json')==ev['training_receipt_sha256']
                assert sha(e.with_name('predictions.json'))==ev['predictions_sha256']
                assert ev['protocol']['annotations_sha256']==state['annotation_sha256']
                assets.append({'seed':seed,'arm':arm,'checkpoint':ev['checkpoint'],'evaluation_path':str(e),'evaluation_sha256':sha(e)})
        state['assets']=assets; write(out/'preflight.json',state)
        def stage(name,cmd):
            item={'name':name,'started_utc':now(),'command':cmd,'status':'running'}
            state['stages'].append(item); state.update(status='running',phase=name); write(out/'status.json',state)
            print('START',name,flush=True)
            with (out/f'{name}.log').open('x') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            item.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed',finished_utc=now()); write(out/'status.json',state)
            if result.returncode: raise RuntimeError(f'{name} failed; inspect {out/name}.log')
            print('DONE',name,flush=True)
        for subset in (True,False):
            for asset in assets:
                name=('smoke' if subset else 'full')+f"-s{asset['seed']}-{asset['arm']}"
                cmd=[sys.executable,'-u','-m','experiments.p2_stage_diagnostics_5090.depth','--variant',asset['arm'],'--seed',str(asset['seed']),'--checkpoint',asset['checkpoint']['path'],'--baseline-evaluation',asset['evaluation_path'],'--expected-commit',ANCESTOR,'--output',str(out/name),'--batch','2']
                if subset: cmd += ['--limit','8']
                stage(name,cmd)
        stage('analysis',[sys.executable,'-u','-m','experiments.p2_stage_diagnostics_5090.analyze','--root',str(out),'--output',str(out/'analysis')])
        assert all(sha(ROOT/k)==v for k,v in pin['source_sha256'].items())
        assert all(sha(Path(x['checkpoint']['path']))==x['checkpoint']['sha256'] for x in assets)
        state.update(status='completed',phase='finished',finished_utc=now(),frozen_sources_unchanged=True,checkpoints_unchanged=True)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}',finished_utc=now()); raise
    finally: write(out/'status.json',state)

if __name__=='__main__':
    with open('/tmp/zhj-uav-5090-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
