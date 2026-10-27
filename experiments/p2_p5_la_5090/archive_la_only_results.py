"""Archive completed LA-only 150e and four-arm 5090 comparison.

Collection uses server artifacts. Rendering and verification use only tracked
text and Python's standard library; no GPU, weights or dataset are required.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from experiments.decoder_budget_5090.report_decoder_screen_5090 import reduction

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'docs/experiment_reports/2026-10-27_5090_la_only_150e_results'
LAUNCH=ROOT/'docs/experiment_reports/2026-10-27_5090_la_only_150e_launch'
DEPTH=ROOT/'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw'
JOINT=ROOT/'docs/experiment_reports/2026-10-27_5090_c2_la_150e_results/raw'
PRE=ROOT/'docs/experiment_reports/2026-10-26_5090_p2_p5_la_preflight/raw/benchmark.json'
BASE_REPORT=ROOT/'reports/5090/experiments.json'
BASE_CAT=ROOT/'docs/experiment_reports/2026-10-26_5090_fresh_decoder_results/raw/category_ap.csv'
ARMS=('c0','c2','la','c2-la')
PAIRS={'c0-la':{'c2':'c0','c2-la':'la'},'c2-joint':{'c2':'c2','c2-la':'c2-la'}}
SHAS={'c0':'adeaec17109a3b3ac6b8a6e521fb6cc3c2bdb3cd124ad6eedecc930d35a03d2e',
      'c2':'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6',
      'c2-la':'1a3866ebc5246748d8c359acb30129baa688566c41cfe3be5ce6a05baab4833f'}


def read(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    path=path.resolve();return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def collect(args):
    root=args.root.resolve();raw=args.output/'raw';raw.mkdir(parents=True,exist_ok=False)
    train_dir=root/'5090-la-only-150e-s0'
    status=read(root/'status.json');training=read(train_dir/'experiment.json');evaluation=read(root/'evaluation/summary.json')
    assert status['status']==training['status']==evaluation['status']=='completed'
    assert status['source_sha256']==training['source_sha256']==evaluation['training_source_sha256']
    compact_status=copy.deepcopy(status);compact_status.pop('source_sha256')
    compact_eval=copy.deepcopy(evaluation);compact_eval.pop('training_source_sha256')
    timing=read(root/'benchmark/assessment.json');compact_timing=copy.deepcopy(timing)
    for item in compact_timing['conditions'].values():
        for block in item.get('blocks',[]):block.pop('wall_ms',None)
    write(raw/'status.json',compact_status);write(raw/'training.json',training);write(raw/'evaluation.json',compact_eval)
    write(raw/'benchmark.json',compact_timing)
    (raw/'training_curve.csv').write_bytes((train_dir/'results.csv').read_bytes())
    health=[json.loads(line) for line in (train_dir/'health.jsonl').read_text().splitlines()]
    keys=list(dict.fromkeys(k for row in health for k in row))
    write_csv(raw/'training_health.csv',[{k:row.get(k) for k in keys} for row in health])
    files=[root/'status.json',train_dir/'experiment.json',train_dir/'results.csv',train_dir/'health.jsonl',
           root/'evaluation/summary.json',root/'evaluation/predictions.json',root/'benchmark/assessment.json']
    files.extend(Path(v['path']) for v in training['checkpoints'].values())
    for item in status['phases']:files.append(Path(item['log']))
    for pair,expected in PAIRS.items():
        view=read(root/('fp16-benchmark-view-'+pair+'.json'))
        assert view['actual_arm_by_alias']==expected
        path=root/('fp16-evaluation-'+pair)/'summary.json'
        summary=read(path);summary['actual_arm_by_alias']=expected
        write(raw/('fp16-'+pair+'.json'),summary)
        files.extend([path,root/('fp16-benchmark-view-'+pair+'.json')])
        files.extend(Path(v['predictions']['path']) for v in summary['models'].values())
    for name in ('replay.json','native-c0-categories.json'):
        path=args.replay_dir/name
        write(raw/name,read(path));files.append(path)
    files.extend([ROOT/'runs/diagnostics/5090-depth-eval-c0-20261025-final/d3-predictions.json',
                  ROOT.parent/'detrscope/src/detrscope/evaluation/coco.py'])
    benchmark_log=(root/'logs/benchmark-four-trained.log').read_text(errors='replace')
    write(raw/'validation.json',{'server_only':[artifact(p) for p in files],
          'archive_timing_policy':'All CUDA samples retained; raw wall samples remain on server; block and pooled wall summaries retained.',
          'fp16_engine_warning_counts':{'clipped_half_constants':benchmark_log.count('contains out-of-range weights when cast to Half'),
                                        'fp16_layernorm':benchmark_log.count('Detected layernorm nodes in FP16')},
          'source_note':'LA-only and C2+LA have identical shared/P5 initial state; older C0/C2 are same seed/data/recipe historical checkpoints, not simultaneous reruns.',
          'fp16_aliases':PAIRS,'fp16_replay':'Four exact saved predictions independently replayed with DETRScope COCOeval.'})


def inputs(archive):
    raw=archive/'raw'
    return (read(raw/'status.json'),read(raw/'training.json'),read(raw/'evaluation.json'),read(raw/'benchmark.json'),
            {p:read(raw/('fp16-'+p+'.json')) for p in PAIRS},read(raw/'replay.json'),read(raw/'native-c0-categories.json'),
            read_csv(raw/'training_curve.csv'),read_csv(raw/'training_health.csv'))


def validate(archive):
    status,train,ev,bench,fp16,replay,c0cats,curve,health=inputs(archive)
    assert status['status']==train['status']==ev['status']=='completed' and not status['warnings']
    assert all(p['returncode']==0 for p in status['phases'])
    assert status['new_training']==['la'] and not status['existing_training_repeated']
    assert train['variant']=='la' and train['role']=='fixed_budget_screen' and train['seed']==0 and train['fraction']==1 and train['completed_epochs']==150
    assert not train['recipe']['pretrained'] and not train['recipe']['resume']
    assert not train['architecture']['p2_injection'] and train['architecture']['p5_na7_rpb']
    assert (train['architecture']['queries'],train['architecture']['points_per_level'],train['architecture']['decoder_layers'])==(300,4,3)
    assert len(curve)==len(health)==150 and [int(float(r['epoch'])) for r in curve]==[int(r['epoch']) for r in health]==list(range(1,151))
    assert all(math.isfinite(float(v)) for row in curve for v in row.values())
    for row in health:
        assert all(float(row[k])>0 for k in ('p5_qkv_grad_abs','p5_rpb_grad_abs'))
        assert 'p2_first_weight_grad_abs' not in row
        assert all(float(v)==1 for k,v in row.items() if k.endswith('_finite'))
    plan=read(LAUNCH/'raw/plan.json');joint=read(JOINT/'training.json')
    assert train['source_sha256']==plan['source_sha256']
    for name,digest in train['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert all(train['source_sha256'].get(k)==v for k,v in joint['source_sha256'].items())
    for key in ('shared_state_sha256','p5_slot_sha256','r0_full_state_sha256'):
        assert train['initialization'][key]==plan['initialization'][key]==joint['initialization'][key]
    assert train['recipe']==joint['recipe'] and train['data_yaml_sha256']==joint['data_yaml_sha256']==sha(LAUNCH/'raw/data.yaml')
    assert ev['scope']=='full_val' and len(ev['protocol']['image_ids'])==548 and ev['checkpoint']==train['checkpoints']['best']
    assert ev['protocol']['dtype']=='float32' and not ev['protocol']['fused'] and ev['protocol']['formal_max_dets']==[1,10,100]
    assert ev['protocol']['conf']==.001 and ev['protocol']['prediction_max_det']==300 and ev['protocol']['batch']==2
    annotation=ev['protocol']['annotations_sha256'];assert annotation==read(LAUNCH/'raw/asset_audit.json')['coco']['sha256']
    old={arm:read(DEPTH/('depth-eval-'+arm+'.json')) for arm in ('c0','c2')}
    for arm in old:
        assert old[arm]['checkpoint']['sha256']==SHAS[arm] and old[arm]['formal_protocol']['annotation_sha256']==annotation
        assert old[arm]['formal_protocol']['image_ids']==ev['protocol']['image_ids']
    assert all(abs(c0cats['metrics'][k]-v)<1e-12 for k,v in old['c0']['conditions']['d3']['metrics'].items())
    joint_ev=read(JOINT/'evaluation.json');assert joint_ev['checkpoint']['sha256']==SHAS['c2-la']
    assert joint_ev['protocol']['annotations_sha256']==annotation and joint_ev['protocol']['image_ids']==ev['protocol']['image_ids']
    assert math.isclose(statistics.fmean(v['AP'] for v in ev['per_class'].values()),ev['metrics']['AP'],abs_tol=1e-12)
    assert math.isclose(statistics.fmean(v['AP'] for v in c0cats['per_class'].values()),c0cats['metrics']['AP'],abs_tol=1e-12)
    assert bench['status']=='completed' and bench['protocol']['rounds']==20 and bench['protocol']['samples_per_round']==50
    assert len(bench['conditions'])==12 and len(bench['round_order'])==20
    expected={**SHAS,'la':train['checkpoints']['best']['sha256']}
    for order in bench['round_order']:assert set(order)==set(bench['conditions'])
    for arm in ARMS:
        for backend in ('eager_fp32','trt_fp32','trt_fp16'):
            item=bench['conditions'][arm+'/'+backend]
            assert item['status']=='completed' and item['checkpoint']['sha256']==expected[arm] and item['output_check']['all_finite']
            assert len(item['blocks'])==20 and {b['round'] for b in item['blocks']}==set(range(1,21))
            samples=[v for b in item['blocks'] for v in b['cuda_ms']]
            assert len(samples)==1000 and all(v>0 and math.isfinite(v) for v in samples)
            assert math.isclose(statistics.fmean(samples),item['cuda_stats_ms']['mean'],abs_tol=1e-12)
            if backend.startswith('trt'):assert item['changed_input_output']['all_finite']
    assert replay['status']=='completed' and set(replay['models'])==set(ARMS)
    for pair,mapping in PAIRS.items():
        data=fp16[pair]
        assert data['status']=='completed' and data['actual_arm_by_alias']==mapping and data['protocol']['images']==548
        assert data['protocol']['image_ids']==ev['protocol']['image_ids'] and data['protocol']['annotation_sha256']==annotation
        assert data['protocol']['category_ids']==ev['protocol']['category_ids']
        assert data['protocol']['formal_max_dets']==[1,10,100] and data['protocol']['conf']==.001
        assert data['protocol']['prediction_max_det']==300 and data['protocol']['batch']==1
        assert data['first_input_sha256']==bench['input']['input_sha256']
        for alias,arm in mapping.items():
            item=data['models'][alias];condition=bench['conditions'][arm+'/trt_fp16'];check=replay['models'][arm]
            assert item['checkpoint']==condition['checkpoint'] and item['checkpoint']['sha256']==expected[arm]
            engine=next(a for a in condition['engine']['server_artifacts'] if a['path'].endswith('.engine'))
            assert item['engine']==engine and check['engine_sha256']==engine['sha256']
            assert check['checkpoint_sha256']==expected[arm] and check['prediction_sha256']==item['predictions']['sha256']
            assert check['gt_count']==38759 and all(abs(check['metrics'][k]-v)<1e-12 for k,v in item['metrics'].items())
            assert math.isclose(statistics.fmean(v['AP'] for v in item['per_class'].values()),item['metrics']['AP'],abs_tol=1e-12)
    print('Verified 150e fresh LA-only, paired source/data/init, full native+FP16 COCO for four arms, 12000 paired CUDA samples and independent FP16 replays.')


def render(archive,output):
    output.mkdir(parents=True,exist_ok=True)
    status,train,ev,bench,fp16,replay,c0cats,curve,health=inputs(archive)
    old={a:read(DEPTH/('depth-eval-'+a+'.json'))['conditions']['d3']['metrics'] for a in ('c0','c2')}
    native={**old,'la':ev['metrics'],'c2-la':read(JOINT/'evaluation.json')['metrics']}
    deployment={}
    for pair,mapping in PAIRS.items():
        for alias,arm in mapping.items():deployment[arm]=fp16[pair]['models'][alias]
    params=read(PRE)['models']
    rows=[]
    for arm in ARMS:
        n=native[arm];f=deployment[arm]['metrics'];timing=bench['conditions'][arm+'/trt_fp16']
        rows.append({'variant':arm,'parameters':params[arm]['parameters_unfused'],
                     'native_AP':n['AP'],'native_AP50':n['AP50'],'native_AP75':n['AP75'],
                     'native_APs':n['APs'],'native_APm':n['APm'],'native_APl':n['APl'],
                     'trt_fp16_AP':f['AP'],'trt_fp16_APs':f['APs'],
                     'trt_fp16_mean_ms':timing['cuda_stats_ms']['mean'],'trt_fp16_p95_ms':timing['cuda_stats_ms']['p95'],
                     'trt_fp16_forward_fps':1000/timing['cuda_stats_ms']['mean']})
    write_csv(output/'comparison.csv',rows)
    effects=[]
    for metric in ('AP','APs','APm','APl','AP50','AP75'):
        a,b,c,d=[100*native[x][metric] for x in ARMS]
        effects.append({'metric':metric,'LA_without_P2_pp':c-a,'LA_with_P2_pp':d-b,
                        'P2_without_LA_pp':b-a,'P2_with_LA_pp':d-c,'interaction_pp':d-b-c+a})
    write_csv(output/'effects.csv',effects)
    latency=[]
    for backend in ('eager_fp32','trt_fp32','trt_fp16'):
        for before,after in (('c0','la'),('c2','c2-la'),('c0','c2'),('la','c2-la')):
            x,y=[bench['conditions'][arm+'/'+backend] for arm in (before,after)]
            r=reduction(x,y)
            for k in ('latency_reduction_percent','ci95_low','ci95_high'):r[k]=round(r[k],9)
            latency.append({'backend':backend,'before':before,'after':after,'before_ms':x['cuda_stats_ms']['mean'],
                            'after_ms':y['cuda_stats_ms']['mean'],**r})
    write_csv(output/'latency_effects.csv',latency)
    categories={}
    categories['c0']=c0cats['per_class'];categories['la']=ev['per_class'];categories['c2-la']=read(JOINT/'evaluation.json')['per_class']
    source=[r for r in read_csv(BASE_CAT) if r['variant']=='c2-q300-p4-d3']
    categories['c2']={r['category']:{k:float(r[k]) if r[k] else None for k in ('AP','APs','APm','APl')} for r in source}
    names=[r['category'] for r in source];counts={r['category']:r['gt_count'] for r in source}
    classes=[]
    for name in names:
        for arm in ARMS:
            n=categories[arm][name];f=deployment[arm]['per_class'][name]
            classes.append({'category':name,'gt_count':counts[name],'variant':arm,
                            **{'native_'+k:n[k] for k in ('AP','APs','APm','APl')},
                            **{'trt_fp16_'+k:f[k] for k in ('AP','APs','APm','APl')}})
    write_csv(output/'category_metrics.csv',classes)
    reports=read(BASE_REPORT)['records'];curves={}
    for arm in ('c0','c2'):
        curves[arm]=next(r['curve'] for r in reports if r.get('checkpoint_sha256',{}).get('best')==SHAS[arm])
    curves['la']=curve;curves['c2-la']=read_csv(JOINT/'training_curve.csv')
    write_csv(output/'training_curves.csv',[{'variant':arm,**r} for arm in ARMS for r in curves[arm]])
    gain=100*(float(curve[149]['metrics/mAP50-95(B)'])-float(curve[124]['metrics/mAP50-95(B)']))
    best=max(curve,key=lambda r:float(r['metrics/mAP50-95(B)']))
    peak=max(float(r['train_peak_allocated_mib']) for r in health);hours=sum(float(r['train_seconds']) for r in health)/3600
    ap=next(r for r in effects if r['metric']=='AP');aps=next(r for r in effects if r['metric']=='APs')
    l0=next(r for r in latency if r['backend']=='trt_fp16' and r['before']=='c0' and r['after']=='la')
    l2=next(r for r in latency if r['backend']=='trt_fp16' and r['before']=='c2' and r['after']=='c2-la')
    lp=next(r for r in latency if r['backend']=='trt_fp16' and r['before']=='la' and r['after']=='c2-la')
    warnings=read(archive/'raw/validation.json')['fp16_engine_warning_counts']
    lines=['# LA-only从零150轮：5090四组对照结果','',
           '分支 `5090-detrscope-query-budget`。唯一新训LA-only；已有C0、C2、C2＋LA训练与权重均复用。全部150轮、原生完整COCO评估、四模型同场测速及四引擎FP16完整COCO评估于2026-10-27 18:03（北京时间）完成。','',
           '**本次seed0、150轮下，LA-only没有优于C0：原生AP低0.350点，TensorRT FP16延迟高6.96%。** 给LA加入P2后原生AP增加0.715点，但仍低于原C2。','',
           '## 四组同协议比较','',
           '精度为完整548张验证图、38759个标注的正式COCO bbox AP，表中乘100；conf=.001，每query取最高类、最多300预测、不做NMS/坐标舍入，COCO maxDets=[1,10,100]。原生未融合FP32为batch2，TensorRT FP16引擎为batch1，输入输出为FP32且引擎允许混合精度。延迟为训练后四模型同场GPU常驻batch1、640纯forward，不含预处理/传输/后处理。','',
           '| 模型 | 原生FP32 AP | 原生APs | TensorRT FP16 AP | TensorRT FP16 ms |','|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['variant']} | {100*row['native_AP']:.3f} | {100*row['native_APs']:.3f} | {100*row['trt_fp16_AP']:.3f} | {row['trt_fp16_mean_ms']:.4f} |")
    lines += ['', 'TensorRT FP16四个实际引擎均在相同548图上完成正式AP评估；四份未舍入预测由独立DETRScope COCO评估器重放，12项指标误差均小于1e-12。[原始FP16结果和别名映射](raw/fp16-c0-la.json) · [另一对](raw/fp16-c2-joint.json) · [独立重放](raw/replay.json)。','',
              '## 改动的实际效果','',
              f"- **单独加LA（C0→LA）**：原生AP {ap['LA_without_P2_pp']:+.3f}点，APs {aps['LA_without_P2_pp']:+.3f}点；FP16延迟上升{-l0['latency_reduction_percent']:.2f}%（{l0['before_ms']:.4f}→{l0['after_ms']:.4f} ms）。",
              f"- **已有P2时加LA（C2→C2＋LA）**：原生AP {ap['LA_with_P2_pp']:+.3f}点，APs {aps['LA_with_P2_pp']:+.3f}点；FP16延迟上升{-l2['latency_reduction_percent']:.2f}%。",
              f"- **LA上加P2（LA→C2＋LA）**：原生AP {ap['P2_with_LA_pp']:+.3f}点，APs {aps['P2_with_LA_pp']:+.3f}点；FP16延迟上升{-lp['latency_reduction_percent']:.2f}%。",
              f"- **P2与LA的交互**：AP交互项 {ap['interaction_pp']:+.3f}点、APs交互项 {aps['interaction_pp']:+.3f}点，计算式为组合−C2−LA＋C0。这是单seed点估计，不能视作统计显著。",'',
              'LA-only的大目标APl比C0高3.499点，但APm与总体AP下降；组合相对C2的大目标APl下降3.474点。分类别看，LA-only相对C0有8类下降、2类上升：bus下降1.752点，自行车和van分别增加0.577、0.602点；bus仅有251个验证标注，单seed下不应过度解释。不同尺度及类别的反应并不一致，无法仅凭总体AP推断原因。[完整文本表](category_metrics.csv)保留四组原生与FP16结果。','',
              'FP16延迟95%区间按20个配对轮次重采样，仅描述本次计时波动，不覆盖重新构建TensorRT引擎、其他图片或设备差异。此前单独C2对组合测速约慢8.20%，本次四模型同场约慢5.85%；两次使用独立构建的引擎与独立测量，不能把某个百分比当成跨构建的固定值。两次方向一致。[完整配对差及区间](latency_effects.csv)。','',
              f"TensorRT构建日志记录{warnings['clipped_half_constants']}条FP16常量裁剪和{warnings['fp16_layernorm']}条LayerNorm相关警告；上述FP16精度是这些实际引擎在完整验证集上的测量。",'',
              '## 训练与有效范围','',
              f"LA-only从零训练150轮、seed0、batch8、640、FP32、AdamW及原配方；训练计算累计{hours:.2f}小时，PyTorch峰值allocated {peak:.0f} MiB。最佳内部mAP出现在第{int(float(best['epoch']))}轮；125→150轮增加{gain:.3f}点。P5 QKV/RPB梯度全程非零，检测框更新有限。",'',
              '历史C0/C2/C2＋LA是独立完成的150轮训练，没有为本次对照重跑。共同数据、配方、seed及随机初值已核验；由于代码执行/训练存在非确定性且只有单seed，效应和交互仅为本次固定预算估计，尚不证明充分收敛。','',
              '![训练曲线与TensorRT FP16精度延迟](figures/summary.svg)','',
              '目前实际权衡下，C2在本次FP16测量中比LA-only和C2＋LA同时具有更高AP和更低延迟；继续使用C2更稳妥。是否存在其他训练策略或更快局部注意力实现，需要另行实验。','',
              '## 可读证据与复现','',
              '- [四模型AP/参数/时延](comparison.csv)、[效应和交互](effects.csv)、[完整训练曲线](training_curves.csv)、[150轮健康](raw/training_health.csv)。',
              '- [LA原生COCO和分类别](raw/evaluation.json)、[训练后12000次CUDA计时](raw/benchmark.json)、[FP16两对结果及显式别名](raw/fp16-c0-la.json)、[另一对](raw/fp16-c2-joint.json)、[独立重放](raw/replay.json)。',
              '- [最终训练/权重路径及SHA](raw/training.json)、[执行状态](raw/status.json)、[服务器文件路径/大小/SHA及构建警告](raw/validation.json)。权重、预测、导出模型和完整日志仅服务器保存。',
              '- [启动计划与复现](../2026-10-27_5090_la_only_150e_launch/README.md)、[既有C2＋LA结果](../2026-10-27_5090_c2_la_150e_results/README.md)。','',
              '从Git检出后，可不访问GPU、数据集或服务器权重运行：','',
              '```bash','python3 -I docs/experiment_reports/2026-10-27_5090_la_only_150e_results/verify.py',
              'python3 -m experiments.p2_p5_la_5090.archive_la_only_results --output /tmp/5090-la-final-replay',
              '# 可在有matplotlib的环境中加 --plots 重建SVG。','```','']
    (output/'REPORT.md').write_text('\n'.join(lines))


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt']='5090-la-only-150e-20261027'
    curves=read_csv(output/'training_curves.csv');rows={r['variant']:r for r in read_csv(output/'comparison.csv')}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors={'c0':'#3565a4','c2':'#2f8a69','la':'#dd7138','c2-la':'#9254a6'}
    for arm in ARMS:
        selected=[r for r in curves if r['variant']==arm]
        axes[0].plot([float(r['epoch']) for r in selected],[100*float(r['metrics/mAP50-95(B)']) for r in selected],label=arm.upper(),color=colors[arm])
        row=rows[arm];x=float(row['trt_fp16_mean_ms']);y=100*float(row['trt_fp16_AP'])
        axes[1].scatter(x,y,color=colors[arm],s=45,label=arm.upper())
    axes[0].set(xlabel='Epoch',ylabel='Internal mAP50-95 (%)',title='150 epochs, seed 0')
    axes[1].set(xlabel='TensorRT FP16 forward latency (ms)',ylabel='TensorRT FP16 COCO AP (%)',title='Same trained engines')
    for axis in axes:axis.grid(alpha=.25);axis.legend(fontsize=8)
    target=output/'figures/summary.svg';target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,metadata={'Date':None});target.write_text('\n'.join(s.rstrip() for s in target.read_text().splitlines())+'\n');plt.close(fig)


def verify_manifest(archive):
    for line in (archive/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    for doc in ('README.md','REPORT.md'):
        for target in re.findall(r'\]\(([^)]+)\)',(archive/doc).read_text()):
            if not target.startswith(('http:','https:')):assert (archive/target.split('#',1)[0]).exists(),target
    print('Verified tracked-only manifest and report links.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=DEFAULT);p.add_argument('--output',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--replay-dir',type=Path)
    p.add_argument('--verify',action='store_true');p.add_argument('--plots',action='store_true');args=p.parse_args()
    if args.root:
        if not args.output or not args.replay_dir:p.error('collect needs --output and --replay-dir')
        collect(args);args.archive=args.output
    validate(args.archive)
    if args.verify:verify_manifest(args.archive)
    else:
        target=args.output or args.archive;render(args.archive,target)
        if args.plots:plots(target)
