"""One-time server bootstrap: vendor the 1.3 MB author backend into Git."""
from pathlib import Path
import hashlib,json,shutil,yaml

REPO=Path('/home/zjnu/文档/ZHJ_WorkSpace/ultralytics')
OLD=REPO.parent/'reproductions/rtdetr-r18-c2-author-recipe-20261028'
EXP=REPO/'experiments/c2_channel_author_20261105'
backend=EXP/'backend/ultralytics'
backend.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(OLD/'ultralytics',backend,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files={str(p.relative_to(backend)):hashlib.sha256(p.read_bytes()).hexdigest() for p in backend.rglob('*') if p.is_file()}
(EXP/'author_backend_original_manifest.json').write_text(json.dumps(files,indent=2)+'\n')
shutil.copyfile(EXP/'gate.py',backend/'nn/modules/p2_channel_gate.py')
p=backend/'nn/tasks.py';s=p.read_text().replace('from ultralytics.nn.modules import *','from ultralytics.nn.modules import *\nfrom ultralytics.nn.modules.p2_channel_gate import P2GateConcat',1).replace('elif m is Concat:', 'elif m in {Concat, P2GateConcat}:',1);p.write_text(s)
cfg=EXP/'configs';cfg.mkdir()
d=yaml.safe_load((OLD/'rtdetr-resnet18-c2.yaml').read_text())
(cfg/'c2.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
for arm in ['scalar','channel']:
    c=yaml.safe_load(yaml.safe_dump(d));assert c['head'][10][2]=='Concat';c['head'][10][2]='P2GateConcat';c['head'][10][3]=[1,128,arm]
    (cfg/f'{arm}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
data=yaml.safe_load((REPO/'configs/visdrone-local.yaml').read_text());data.pop('test',None)
(cfg/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
shutil.copyfile(REPO/'LICENSE',EXP/'backend/LICENSE')
(EXP/'backend/PROVENANCE.md').write_text('''# Frozen author engine

SO-DETR author upstream: https://github.com/ValiantDiligent/SO_DETR
Upstream commit: 148d5972bbaec298b3975ca00c00d91abbc6657c.
Ultralytics 8.0.201, AGPL-3.0. This is the existing B0/B1 reconstructed-baseline engine,
copied from rtdetr-r18-c2-author-recipe-20261028. Ordinary IoU/GIoU restored;
unrelated broken YOLOv10 loader alias removed. New changes: p2_channel_gate.py
and parser import/Concat channel accounting in nn/tasks.py only.
Original backend file hashes are in ../author_backend_original_manifest.json.
All backend source and model/data configs are tracked by Git, not just server-only snapshots.
''')
print('VENDORED',len(files),'files',sum(p.stat().st_size for p in backend.rglob('*') if p.is_file()))
