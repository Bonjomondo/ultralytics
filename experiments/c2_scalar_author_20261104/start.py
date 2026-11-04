"""Start exactly one author-recipe scalar job after checked smoke verification."""
from pathlib import Path
import json,shlex,subprocess,sys,hashlib

ROOT=Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions/c2-scalar-author-20261104')
REPORT=Path('/home/zjnu/文档/ZHJ_WorkSpace/ultralytics/docs/experiment_reports/2026-11-04_scalar_author350')
assert json.loads((ROOT/'smoke_verification.json').read_text())['status']=='passed'
assert not (ROOT/'controller_status.json').exists(),'Already launched: inspect instead of duplicating'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
manifest=json.loads((ROOT/'source_manifest.json').read_text())
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
env={'PYTHONPATH':str(ROOT),'PYTHONFAULTHANDLER':'1','YOLO_AUTOINSTALL':'false','OMP_NUM_THREADS':'8',
     'MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg','WANDB_MODE':'disabled','COMET_MODE':'DISABLED'}
cmd=['env',*[f'{k}={v}' for k,v in env.items()],sys.executable,'-u',str(ROOT/'controller.py')]
script=ROOT/'launch.sh';script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(cmd)+' > '+shlex.quote(str(ROOT/'controller.log'))+' 2>&1\n')
tmux='/home/zjnu/anaconda3/bin/tmux';session='c2-scalar-author-20261104'
assert subprocess.run([tmux,'has-session','-t',session],capture_output=True).returncode!=0
subprocess.run([tmux,'new-session','-d','-s',session,'bash',str(script)],check=True)
receipt={'tmux':session,'root':str(ROOT),'status':str(ROOT/'controller_status.json'),'run':str(ROOT/'runs/c2-scalar-author-350e-s0'),
         'phases':['train','predict','coco','benchmark','final_compare'],'training':'350 cap, patience40, fresh seed0; historical B0 and C2 as controls',
         'source_manifest_sha256':hashlib.sha256((ROOT/'source_manifest.json').read_bytes()).hexdigest()}
(REPORT/'launch.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(receipt)
