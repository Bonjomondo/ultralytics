"""Reproduce the author backend in a new directory; preserve all historical experiments."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

BASE = Path('/home/zjnu/文档/ZHJ_WorkSpace/reproductions')
OLD = BASE / 'rtdetr-r18-c2-author-recipe-20261028'
ROOT = BASE / 'c2-scalar-author-20261104'
REPO = Path('/home/zjnu/文档/ZHJ_WorkSpace/ultralytics')
SOURCE = REPO / 'experiments/c2_scalar_author_20261104'
ROOT.mkdir(exist_ok=False)
shutil.copytree(OLD / 'ultralytics', ROOT / 'ultralytics',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pt', '.git', 'runs'))
for name in ['data.yaml','data-smoke.yaml','dataset_audit.json','baseline_reference.json','baseline_results.csv',
             'rtdetr-resnet18-baseline.yaml','rtdetr-resnet18-c2.yaml','dataset_current_verification.json',
             'paired_initialization_verification.json','architecture_comparison.json','coco_eval.py']:
    shutil.copyfile(OLD / name, ROOT / name)
for name in ['safe_train.py','preflight.py','controller.py','benchmark.py']:
    shutil.copyfile(SOURCE / name, ROOT / name)
module = '''"""Scalar P2 gate: preserve [P4up,P2down,P3] concat and identity initialization."""
import torch
from torch import nn

class ScalarP2Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, xs):
        return torch.cat([xs[0], xs[1] * (2 * self.logit.sigmoid()), xs[2]], self.dimension)
'''
(ROOT / 'ultralytics/nn/modules/scalar_gate.py').write_text(module)
p=ROOT/'ultralytics/nn/tasks.py';original=p.read_text();modified=original.replace('from ultralytics.nn.modules import *','from ultralytics.nn.modules import *\nfrom ultralytics.nn.modules.scalar_gate import ScalarP2Concat',1).replace('elif m is Concat:', 'elif m in {Concat, ScalarP2Concat}:',1);p.write_text(modified)
(SOURCE/'backend_patch.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/ultralytics/nn/tasks.py',tofile='b/ultralytics/nn/tasks.py',n=0)))
(SOURCE/'scalar_gate.py').write_text(module)
cfg=(ROOT/'rtdetr-resnet18-c2.yaml').read_text();assert '[[16, 17, 5], 1, Concat, [1]]' in cfg
cfg=cfg.replace('[[16, 17, 5], 1, Concat, [1]]','[[16, 17, 5], 1, ScalarP2Concat, [1]]')
(ROOT/'rtdetr-resnet18-scalar.yaml').write_text(cfg)
(SOURCE/'rtdetr-resnet18-scalar.yaml').write_text(cfg)
s=(OLD/'train_c2.py').read_text().replace("NAME = 'rtdetr-r18-c2-matched-b0-350e-s0'", "NAME = 'c2-scalar-author-350e-s0'")
s=s.replace('class PairedTrainer(RTDETRTrainer):','from safe_train import SafeTrainer\n\n\nclass PairedTrainer(SafeTrainer):')
s=s.replace("'rtdetr-resnet18-c2.yaml'", "'rtdetr-resnet18-scalar.yaml'")
s=s.replace("== 20135288", "== 20135289")
s=s.replace("not k.startswith('model.17.')", "not k.startswith(('model.17.', 'model.18.'))")
s=s.replace("'candidate_parameters':", "'gate_initial_multiplier': 1.0,\n            'candidate_parameters':")
s=s.replace("'train_c2.py'", "'train_scalar.py'")
s=s.replace("'benchmark.py', 'finish.py', 'run.sh', 'architecture_comparison.json'", "'benchmark.py', 'safe_train.py', 'controller.py', 'preflight.py', 'architecture_comparison.json'")
s=s.replace("state.update(status='training',", "state.update(gate_multiplier=float((2*trainer.model.model[18].logit.sigmoid()).detach()), gate_grad_abs=getattr(trainer,'gate_grad_abs',None), status='training',")
s=s.replace("model.add_callback('on_model_save', snapshot)","# Preserve best/last via SafeTrainer; snapshots are retained at the run boundary.")
s=s.replace("parser.add_argument('--resume', type=Path)","parser.add_argument('--resume', type=Path)\n    # The legacy framework resumes EMA by default; require a separately audited raw-state restore.\n")
s=s.replace("args = parser.parse_args()", "args = parser.parse_args()\n    if args.resume:\n        raise RuntimeError('Implicit EMA resume disabled; use an audited raw-state recovery from preserved FP32 checkpoints')")
(ROOT/'train_scalar.py').write_text(s);(SOURCE/'train_scalar.py').write_text(s)
s=(OLD/'export_predictions.py').read_text().replace('workers=2','workers=0')
(ROOT/'export_predictions.py').write_text(s);(SOURCE/'export_predictions.py').write_text(s)
files=[p for p in ROOT.rglob('*') if p.is_file()]
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
identity={'root':str(ROOT),'author_engine_source':str(OLD),'patched_files':['ultralytics/nn/tasks.py','ultralytics/nn/modules/scalar_gate.py'],
          'baseline_AP':27.773305535527815,'C2_AP':28.72234319624046,'new_parameters':1,
          'training':'fresh seed0, batch4/nbs4,640,FP32,AdamW1e-4,author warmup2000 batches,workers8,350 epoch cap/patience40',
          'checkpoint_change':'CPU FP32 raw/EMA/optimizer/RNG with stdlib pickle and atomic replacement; math unchanged',
          'limits':'Historical B0/B1 initial full checkpoints were not archived; seed constructor replay is verified against B1 common tensor receipt, not original archived full initial state.'}
(SOURCE/'experiment_spec.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
print(ROOT)
