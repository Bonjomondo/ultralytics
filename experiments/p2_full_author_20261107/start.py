"""Freeze committed scripts plus the tracked author backend and launch one queue."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from common import BACKEND, BRANCH, EXP, REPO, REPORT, RUNS, now, sha, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-controller',action='store_true')
    args=parser.parse_args()
    tmux='/home/zjnu/anaconda3/bin/tmux';session='p2-full-author-20261107'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
    assert subprocess.run([tmux,'has-session','-t',session],capture_output=True).returncode!=0,'queue alive'
    RUNS.mkdir(exist_ok=True)
    frozen=RUNS/'source'
    if not args.resume_controller:
        assert not (RUNS/'queue_status.json').exists(),'queue already started'
        assert not frozen.exists()
        assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()==BRANCH
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'commit reviewed source first'
        for name in ['preflight.json','smoke_verification.json','resume_verification.json','dense_memory_probe.json','tiny_recall_verification.json']:
            assert json.loads((REPORT/name).read_text())['status']=='passed',name
        validated=json.loads((REPORT/'smoke_verification.json').read_text())['validated_source_hashes']
        actual={str(path.relative_to(EXP)):sha(path) for path in sorted(EXP.rglob('*'))
                if path.is_file() and path.suffix in ['.py','.yaml'] and '__pycache__' not in path.parts}
        assert validated==actual,'Source changed after short-run verification'
        mapping={}
        for source_root,prefix in [(EXP,Path('.')),(BACKEND,Path('backend'))]:
            paths=subprocess.check_output(['git','ls-files','-z','--',str(source_root.relative_to(REPO))],cwd=REPO).decode().split('\0')
            for name in filter(None,paths):
                source=REPO/name
                relative=prefix/source.relative_to(source_root)
                target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                mapping[str(relative)]=sha(target)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        write(RUNS/'source_identity.json',{'commit':commit,'branch':BRANCH,'manifest':mapping})
        write(REPORT/'source_identity.json',{'commit':commit,'snapshot':str(frozen),
              'manifest_sha256':sha(RUNS/'source_identity.json'),'files':len(mapping)})
    else:
        assert json.loads((RUNS/'queue_status.json').read_text())['status']!='complete'
    config=RUNS/'yolo_config';config.mkdir(exist_ok=True)
    shutil.copyfile(Path.home()/'.config/Ultralytics/Arial.ttf',config/'Arial.ttf')
    env={'PYTHONPATH':f'{frozen}/backend:{frozen}','AUTHOR_P2_REPO':str(REPO),
         'AUTHOR_P2_MANIFEST_SHA':sha(RUNS/'source_identity.json'),'PYTHONFAULTHANDLER':'1',
         'YOLO_CONFIG_DIR':str(config),'YOLO_AUTOINSTALL':'false','OMP_NUM_THREADS':'8',
         'MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg',
         'WANDB_MODE':'disabled','COMET_MODE':'DISABLED'}
    log=RUNS/f'controller-{__import__("time").time_ns()}.log'
    launcher=RUNS/'launch.sh'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(frozen))+'\nexec '+
        shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],sys.executable,'-u',str(frozen/'controller.py')])+
        ' > '+shlex.quote(str(log))+' 2>&1\n')
    subprocess.run([tmux,'new-session','-d','-s',session,'bash',str(launcher)],check=True)
    write(REPORT/'launch.json',{'started_utc':now(),'session':session,'controller_log':str(log),
          'environment':env,'budget':'three arms, each fresh350 seed1; no automatic extra experiments'})
    print('STARTED',session)


if __name__=='__main__':
    main()
