"""Isolated author-engine entry and immutable experiment receipts."""
from pathlib import Path
import os,sys,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
ENGINE=ROOT/'author_engine'
DATA=REPO.parent/'reproductions/rtdetrv3_dense_supervision_5090/data'
RUNS=REPO/'runs/rtdetrv3_dense_supervision_5090/author-stageA-s0-20261105-r1'
REPORT=REPO/'docs/experiment_reports/2026-11-05_5090_dense_supervision_author'
PYTHON='/home/ubuntu/miniconda3/envs/UAV_CV/bin/python'
sys.path.insert(0,str(ENGINE))
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(RUNS/'.yolo'))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('COMET_MODE','DISABLED')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
    tmp.replace(p)

def digest(state,keys=None):
    h=hashlib.sha256()
    for k in sorted(state if keys is None else keys):
        t=state[k].detach().cpu().contiguous()
        h.update(k.encode()+b'\0'+str(t.dtype).encode()+b'\0'+str(tuple(t.shape)).encode()+b'\0')
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def source_hashes():
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in {'.py','.yaml','.json','.md','.sh'} and '__pycache__' not in p.parts}

def verify_pin():
    expected=json.loads((RUNS/'source_pin.json').read_text())
    assert expected==source_hashes(),'Frozen study source has changed'

def environment():
    import torch,ultralytics,platform
    assert Path(sys.executable)==Path(PYTHON),sys.executable
    assert Path(ultralytics.__file__).is_relative_to(ENGINE),ultralytics.__file__
    return dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),ultralytics=ultralytics.__file__,gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())

def recipe(arm,smoke=False):
    import yaml
    args=yaml.safe_load((ROOT/'reference/c0_args.yaml').read_text())
    for k in ['task','mode','save_dir','model','data','project','name','resume']:args.pop(k,None)
    # 350-epoch recipe is held constant; stop at 150 with optimizer retained.
    args.update(model=str(ROOT/'configs'/('c0.yaml' if arm.startswith('a0') or arm.startswith('a2') else 'c2.yaml')),data=str(DATA/('smoke.yaml' if smoke else 'data.yaml')),project=str(RUNS/('smoke' if smoke else 'training')),name=arm,exist_ok=False,epochs=1 if smoke else 350,patience=0 if smoke else 40,workers=2 if smoke else 8,plots=False if smoke else True)
    return args
