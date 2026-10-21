"""Verify finished evidence and refresh compact manifest; no model execution."""
import os
import re
from pathlib import Path
import subprocess
import sys
from detrscope.core.io import read_json, write_json, sha256_file
from source_adapter import PinnedR18NAAdapter

p=Path(__file__).resolve().parent;repo=p.parents[2]
raw=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/detrscope_20261021'
assert read_json(p/'completion.json')['status']=='completed'
for condition in ('baseline','probe_sample','identity_sample','zero_d3_sample'):
    m=read_json(raw/condition/'manifest.json');assert m['status']=='completed'
    for b in m['batches']:
        assert sha256_file(raw/condition/b['path'])==b['sha256']
        assert b['identity_passed'] and b['restoration_passed']
    pred=raw/condition/'predictions.json';coco=read_json(raw/condition/'coco.json')
    binding={'record_manifest_sha256':sha256_file(raw/condition/'manifest.json'),
             'predictions_file_sha256':sha256_file(pred),'coco_canonical_predictions_sha256':coco['predictions_sha256'],
             'conversion':'Recorded final-stage query boxes/probabilities -> native Predictor-compatible argmax/conf>0.001/sort/top300/pixel xywh',
             'boundary_check':'Native Dataset/Predictor exact preprocessing and native Predictor postprocessing, first 2 real images; final record/native eager output exact on every collected image'}
    if condition=='baseline':bindings={}
    bindings[condition]=binding
write_json(p/'prediction_binding.json',bindings)
r=read_json(p/'offline_replay.json')
for name in ('report.md','report.html'):
    assert sha256_file(raw/'baseline'/name)==sha256_file(raw/'offline_replay'/name)
    r[name+'_sha256_equal']=True
write_json(p/'offline_replay.json',r)
complete=read_json(p/'completion.json');complete['offline_replay']=r;write_json(p/'completion.json',complete)
c=read_json(p/'compatibility.json')
for condition in ('baseline','probe_sample'):
    c['conditions'][condition]['identity_intervention_passed_all']=None
write_json(p/'compatibility.json',c)
weights=repo/'runs/detect/r18_na_ablation/rtdetr-r18-r1-150e-s0/weights'
assert sha256_file(weights/'best.pt')==sha256_file(weights/'last.pt')==c['provenance']['checkpoint_sha256']
pin=read_json(p/'source_pin.json')
assert {str(f.relative_to(repo)):sha256_file(f) for f in sorted((repo/'ultralytics').rglob('*.py'))}==pin['source_sha256']
write_json(p/'analysis_runtime.json',{'successful_full_analysis_and_replay':True,'python_executable':sys.executable,
    'analysis_cpu_affinity':[24,25,26,27],'python_allocator':'debug','python_flags':'-X dev',
    'runtime_issue_root_cause':'Not established. Two unbound attempts raised TypeError; debug/dev unbound attempt segfaulted. Affinity-limited full analysis and replay completed with exact SHA/statistics/COCO equality.',
    'failure_logs':['logs/detrscope_r18_full_20261021.log','logs/detrscope_r18_offline_20261021.log','logs/detrscope_r18_debug_20261021.log'],
    'successful_analysis_log':'logs/detrscope_r18_affinity_20261021.log','native_validator_log':'logs/detrscope_r18_native_formal_20261021.log',
    'weights_unchanged':True,'all_model_source_sha256_unchanged':True,'raw_path':str(raw)})
subprocess.run([sys.executable,str(p/'write_report.py')],check=True)
# Normalize Git-facing CSV text only; immutable raw evidence remains unchanged.
for f in p.glob('*.csv'):
    f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
f=p/'comparison_report.md'
text=f.read_text()
for match in list(re.finditer(r'\[([^]]+)\]\(([^)]+)\)',text)):
    label,target=match.groups()
    if target.startswith(('http','#')) or (p/target).exists():continue
    exported=p/('comparison_'+Path(target).name)
    replacement=f'[{label}]({exported.name})' if exported.exists() else f'`{raw/"zero_d3_comparison"/target}`（仅服务器）'
    text=text.replace(match.group(0),replacement)
f.write_text(text)
# No code or report changes after this index is written.
files={f.name:{'bytes':f.stat().st_size,'sha256':sha256_file(f)} for f in sorted(p.iterdir()) if f.is_file() and f.name!='artifact_index.json'}
write_json(p/'artifact_index.json',{'files':files,'schema':'r18-na.git-evidence.v1','csv_line_endings':'LF in Git exports; raw evidence retains original line endings','raw_output':str(raw),
    'scope':'Source/configuration, metadata, lightweight tables and selected cases; no weights/images/full raw trajectories/logs.'})
print('FINAL VERIFIED',len(files),'files',sum(f['bytes'] for f in files.values()),'bytes')
