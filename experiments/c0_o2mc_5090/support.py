"""Identity checks and four-arm paired summary for the original 150e protocol."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from experiments.c2_o2mc_5090 import run as old
from experiments.c2_o2mc_5090.multiseed_support import check_reference, manifest
from experiments.c2_o2mc_5090.model import build_o2mc
from experiments.r18_p2_sfi.train import MODELS, build_model, tensor_digest

ROOT=old.ROOT
def baseline_path(seed):
    if seed==0:
        return ROOT/'runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1/5090-rtdetr-r18-c0-150e-s0/weights/best.pt'
    return ROOT/f'runs/p2_stability_5090/5090-p2-stability-20261027-r1/training/5090-c0-150e-s{seed}/weights/best.pt'

def paired_initialization(seed,checkpoint):
    import torch
    torch.set_num_threads(4)
    directory=checkpoint.parents[1]
    candidates=[directory/'experiment.json',directory/'experiment_5090.json']
    receipt=json.loads(next(p for p in candidates if p.exists()).read_text())
    assert receipt['status']=='completed' and receipt.get('completed_epochs',receipt.get('current_epoch'))==150
    initial=json.loads((directory/'initialization.json').read_text())
    plain=build_model(str(MODELS['c0']),seed=seed)
    aux=build_o2mc(str(MODELS['c0']),seed=seed)
    native=tensor_digest(plain.state_dict())
    assert native==initial['r0_full_state_sha256']==plain.ablation_initialization['r0_full_state_sha256']
    assert native==aux.ablation_initialization['c2_full_state_sha256']
    assert tensor_digest(aux.model.state_dict())==tensor_digest(plain.model.state_dict())
    expected=receipt['checkpoints']['best']['sha256']
    assert old.sha256_file(checkpoint)==expected
    reference=json.loads((ROOT/'docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed'/f'seed{seed}/results/training.json').read_text())
    recipe=reference['recipe'].copy();recipe.pop('save_period',None)
    historic=receipt['recipe'].copy()
    effective=receipt.get('effective_train_args',{})
    for k in ['fraction','workers','pretrained','resume']:
        if k not in historic:
            value=effective.get(k,receipt.get(k))
            if k=='resume' and value is None:
                assert receipt['initialization'].get('trained_checkpoint_loaded',False) is False
                value=False
            assert value is not None,('Missing actual configuration evidence',k)
            historic[k]=value
    for k in ['save_period']:historic.pop(k,None)
    assert recipe==historic,(recipe,historic)
    return {'seed':seed,'native_full_state_sha256':native,'r0_full_state_sha256':native,
            'baseline_checkpoint_sha256':expected,'historical_initialization':initial,
            'native_state_exact':True,'recipe_matches_c2_aux_except_save_period':True,'aux_seed':4000+seed}

def summarize(output):
    rows=[]; classes=[]
    c2=ROOT/'docs/experiment_reports/2026-11-01_5090_c2_o2mc_multiseed'
    metrics=['AP','AP50','AP75','APs','APm','APl','AR100']
    for seed in (0,1,2):
        here=output/f'seed{seed}/results'
        if not (here/'evaluation.json').exists():continue
        a=json.loads((here/'baseline_evaluation.json').read_text())
        b=json.loads((here/'evaluation.json').read_text())
        c=json.loads((c2/f'seed{seed}/results/baseline_evaluation.json').read_text())
        d=json.loads((c2/f'seed{seed}/results/evaluation.json').read_text())
        assert a['protocol']==b['protocol']==c['protocol']==d['protocol']
        row={'seed':seed}
        for k in metrics:
            delta0=100*(b['metrics'][k]-a['metrics'][k]);delta2=100*(d['metrics'][k]-c['metrics'][k])
            row.update({f'C0_{k}':100*a['metrics'][k],f'C0_aux_{k}':100*b['metrics'][k],
                        f'C2_{k}':100*c['metrics'][k],f'C2_aux_{k}':100*d['metrics'][k],
                        f'delta_C0_{k}_pp':delta0,f'delta_C2_{k}_pp':delta2,f'interaction_{k}_pp':delta2-delta0})
        rows.append(row)
        for name in a['per_class']:
            v=[100*x['per_class'][name]['AP'] for x in [a,b,c,d]]
            classes.append(dict(seed=seed,category=name,C0_AP=v[0],C0_aux_AP=v[1],C2_AP=v[2],C2_aux_AP=v[3],interaction_AP_pp=(v[3]-v[2])-(v[1]-v[0])))
    if rows:
        for filename,rs in [('factorial.csv',rows),('class_factorial.csv',classes)]:
            with (output/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
        stats={k:{'mean':statistics.mean(row[k] for row in rows),'sample_std':statistics.stdev(row[k] for row in rows) if len(rows)>1 else None,'n':len(rows)} for k in rows[0] if k.startswith(('delta_','interaction_'))}
        old.write_atomic(output/'factorial_statistics.json',stats)
    lines=['# 原150轮协议：C0 O2M-C补齐与2×2消融','',f'已完成归档seed：{[row["seed"] for row in rows]}。三组全部完成后才形成完整多seed判断。','',
           '| Seed | C0 | C0+Aux | C2 | C2+Aux | ΔC0 | ΔC2 | 交互项 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+str(row['seed'])+' | '+' | '.join(f'{row[k]:.6f}' for k in ['C0_AP','C0_aux_AP','C2_AP','C2_aux_AP','delta_C0_AP_pp','delta_C2_AP_pp','interaction_AP_pp'])+' |')
    lines+=['','原协议150轮、辅助权重0.5，正式COCO maxDets100；不能与作者350轮alpha1协议混用。交互项=(C2+Aux−C2)−(C0+Aux−C0)，seed数少不作为统计显著性证明。','',
            '[计划](PLAN.md) · [已有C2三seed证据](../2026-11-01_5090_c2_o2mc_multiseed/README.md)']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');manifest(output)
