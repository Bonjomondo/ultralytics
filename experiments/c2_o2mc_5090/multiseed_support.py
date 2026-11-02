"""Paired-seed identity gates and compact, independently verifiable summaries."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from experiments.c2_o2mc_5090 import run as original
from experiments.c2_o2mc_5090.model import build_o2mc
from experiments.p2_p5_la_5090.model import build_model as historical_build
from experiments.r18_p2_sfi.train import MODELS, RECIPE, tensor_digest

ROOT = original.ROOT
REFERENCE = ROOT / 'runs/c2_o2mc_5090/5090-c2-o2mc-20261031-s0-r2'


def check_reference():
    frozen = json.loads((REFERENCE/'plan.json').read_text())
    actual = original.source_hashes()
    changed = [p for p,h in frozen['source_sha256'].items() if actual.get(p)!=h]
    if changed:
        raise RuntimeError('Seed0 frozen source changed: '+repr(changed))
    train = json.loads((REFERENCE/'training/5090-c2-o2mc-150e-s0/experiment.json').read_text())
    recipe = {**RECIPE, 'epochs':150, 'seed':0, 'workers':0, 'fraction':1.,
              'pretrained':False, 'resume':False, 'save_period':-1}
    if train['recipe'] != recipe:
        raise RuntimeError('Training recipe differs from seed0')
    if original.sha256_file(original.ANN)!=frozen['annotations_sha256']:
        raise RuntimeError('Annotations differ from seed0')
    return {'path':str(REFERENCE), 'original_source_files_verified':len(frozen['source_sha256']),
            'plan_sha256':original.sha256_file(REFERENCE/'plan.json'),
            'training_recipe_equal_except_seed':True}


def paired_initialization(seed, checkpoint):
    import torch
    torch.set_num_threads(4)
    receipt_path=checkpoint.parents[1]/'experiment.json'
    receipt=json.loads(receipt_path.read_text())
    init=json.loads((checkpoint.parents[1]/'initialization.json').read_text())
    if receipt['status']!='completed' or receipt['completed_epochs']!=150 or receipt['seed']!=seed:
        raise RuntimeError('Incomplete or wrong-seed historical baseline')
    if original.sha256_file(checkpoint)!=receipt['checkpoints']['best']['sha256']:
        raise RuntimeError('Historical baseline checkpoint SHA mismatch')
    reference=historical_build(str(MODELS['c2']),seed=seed)
    for key in ['r0_full_state_sha256','shared_state_sha256','p2_slot_sha256','p5_slot_sha256']:
        if reference.ablation_initialization[key]!=init[key]:
            raise RuntimeError('Historical initialization identity mismatch: '+key)
    candidate=build_o2mc(str(MODELS['c2']),seed=seed)
    full=tensor_digest(reference.state_dict())
    if full!=candidate.ablation_initialization['c2_full_state_sha256']:
        raise RuntimeError('Historical and new main-path initial weights differ')
    if tensor_digest(reference.model.state_dict())!=tensor_digest(candidate.model.state_dict()):
        raise RuntimeError('Actual native state mismatch')
    old_recipe={k:v for k,v in receipt['recipe'].items() if k!='seed'}
    new_recipe={k:v for k,v in {**RECIPE,'epochs':150,'workers':0,'fraction':1.,'pretrained':False,'resume':False}.items() if k!='seed'}
    if old_recipe!=new_recipe:
        raise RuntimeError('Historical baseline training recipe mismatch')
    return {'seed':seed,'native_full_state_sha256':full,
            'r0_full_state_sha256':init['r0_full_state_sha256'],
            'historical_initialization':init,'baseline_receipt_sha256':original.sha256_file(receipt_path),
            'actual_native_state_equal':True,'baseline_recipe_equal':True,
            'aux_seed':seed+4000,'p2_seed':seed+1000}


def manifest(directory):
    paths=sorted(p for p in directory.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256')
    (directory/'MANIFEST.sha256').write_text(''.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(directory).as_posix()+'\n' for p in paths))


def summarize(output):
    """Use compact archived files only; no server-only checkpoints required."""
    pairs=[]
    for seed in (0,1,2):
        folder=output/f'seed{seed}'/'results'
        if not (folder/'comparison.json').exists(): continue
        a=json.loads((folder/'evaluation.json').read_text())
        b=json.loads((folder/'baseline_evaluation.json').read_text())
        tr=json.loads((folder/'training.json').read_text())
        if a['protocol']!=b['protocol'] or a['status']!='completed' or b['status']!='completed':
            raise RuntimeError('Evaluation identity mismatch')
        if tr['seed']!=seed or tr['status']!='completed' or tr['completed_epochs']!=150:
            raise RuntimeError('Wrong or incomplete archived training')
        if a['checkpoint']['sha256']!=tr['checkpoints']['best']['sha256']:
            raise RuntimeError('Evaluation not bound to best checkpoint')
        if pairs and a['protocol']!=pairs[0]['protocol']:
            raise RuntimeError('Cross-seed evaluation protocols differ')
        pair={'seed':seed,'baseline_AP':100*b['metrics']['AP'],'candidate_AP':100*a['metrics']['AP'],
              **{'delta_'+k+'_pp':100*(a['metrics'][k]-b['metrics'][k]) for k in a['metrics']},
              'protocol':a['protocol']}
        pairs.append(pair)
    fields=['seed','baseline_AP','candidate_AP']+['delta_'+k+'_pp' for k in ('AP','AP50','AP75','APs','APm','APl')]
    with (output/'paired_seeds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pairs)
    stats={k:{'mean':statistics.mean(p[k] for p in pairs),
              'sample_std':statistics.stdev(p[k] for p in pairs) if len(pairs)>1 else None,
              'positive_pairs':sum(p[k]>0 for p in pairs),'n':len(pairs)} for k in fields if k.startswith('delta_')}
    original.write_atomic(output/'paired_statistics.json',stats)
    lines=['# C2 + O2M-C：三 seed 配对实验','',
           f'已归档 seed：{[p["seed"] for p in pairs]}。'+('三组均已完成。' if len(pairs)==3 else '队列尚未全部结束。'),'',
           '| Seed | C2 AP | C2+O2M-C AP | ΔAP pp | ΔAPs pp | ΔAPm pp | ΔAPl pp |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for p in pairs:
        lines.append(f'| {p["seed"]} | {p["baseline_AP"]:.6f} | {p["candidate_AP"]:.6f} | {p["delta_AP_pp"]:+.6f} | {p["delta_APs_pp"]:+.6f} | {p["delta_APm_pp"]:+.6f} | {p["delta_APl_pp"]:+.6f} |')
    if pairs:
        d=stats['delta_AP_pp']
        lines+=['',f'配对 ΔAP 均值 {d["mean"]:+.6f}pp，样本标准差 {d["sample_std"]}pp，正向 {d["positive_pairs"]}/{d["n"]}。']
    lines+=['','正式指标为完整548图、原生未融合FP32的COCO bbox评估，maxDets=[1,10,100]；不是VisDrone官方挑战分数。',
            '固定150轮预算不代表收敛。只比较C2与C2+辅助监督，未运行C0-Aux，不能宣称结构与监督协同；三seed不构成统计显著性证明。','',
            '[配对原始表](paired_seeds.csv) · [配对统计](paired_statistics.json) · [协议和复现](PLAN.md)','',
            '各seed目录保留完整训练曲线、逐轮健康/梯度、正式评估、基线复评、源码/数据/权重身份与SHA。权重、完整预测和大日志只留服务器。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    manifest(output)
