"""Train only missing 5090 LA-only seed0 150e; reuse completed factorial arms.

Persistent controller: data identity -> fresh training -> full native COCO ->
four-checkpoint paired timing -> two full TensorRT FP16 evaluation pairs ->
server-side report. No speed gate, no pretrained model and no repeated training.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NAME = '5090-la-only-150e-s0'
OLD = ROOT / 'runs/decoder_budget_5090/5090-c0c2-baselines-20261024-s0-r1'
JOINT = ROOT / 'runs/p2_p5_la_5090/5090-c2-la-150e-20261026-s0-r1'
CHECKPOINTS = {
    'c0': (OLD / '5090-rtdetr-r18-c0-150e-s0/weights/best.pt', 'adeaec17109a3b3ac6b8a6e521fb6cc3c2bdb3cd124ad6eedecc930d35a03d2e'),
    'c2': (OLD / '5090-rtdetr-r18-c2-150e-s0/weights/best.pt', 'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6'),
    'c2-la': (JOINT / '5090-c2-la-150e-s0/weights/best.pt', '1a3866ebc5246748d8c359acb30129baa688566c41cfe3be5ce6a05baab4833f'),
}


def read(path):
    return json.loads(path.read_text())


def progress(root, state):
    path=root/NAME/'experiment.json'
    training=read(path) if path.exists() else {}
    if training:
        state['completed_epochs']=training.get('completed_epochs',0)
        state['training_status']=training.get('status')
        state['latest_internal_map']=training.get('latest_internal_map')
    lines=['# LA-only：5090从零150轮', '',
           f"状态：{state['status']}；阶段：{state.get('phase')}；训练：{state.get('completed_epochs',0)}/150。", '',
           '仅新训LA-only；不带P2，300Q、4点、3层。已有C0/C2/C2＋LA只读取对照，不重训。',
           '训练后自动进行548图正式AP、四模型同场测速和可用的TensorRT FP16全验证集评估。', '']
    if training.get('best_internal_map') is not None:
        lines.append(f"当前最佳训练内部mAP：{100*training['best_internal_map']:.3f}%，不等于正式COCO AP。")
    (root/'progress.md').write_text('\n'.join(lines)+'\n')


def controller(args):
    from experiments.decoder_budget_5090.train_5090 import git,now,sha256_file,write_atomic
    from experiments.p2_p5_la_5090.model import MODELS,build_model
    from experiments.p2_p5_la_5090.train import source_hashes
    from experiments.r18_p2_sfi.train import tensor_digest
    root=args.root.resolve()
    if not args.deployment_python.is_file():
        raise FileNotFoundError('Missing TensorRT virtualenv interpreter')
    for arm,(path,digest) in CHECKPOINTS.items():
        if not path.is_file() or sha256_file(path)!=digest:
            raise ValueError('Historical checkpoint missing or differs: '+arm)
    root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir()
    shutil.copyfile(args.data,root/'data.yaml')
    state={'schema':'5090.la-only150.queue.v1','status':'running','phase':'prepare','started_utc':now(),
           'root':str(root),'variant':'la','epochs_requested':150,'seed':0,'new_training':['la'],
           'existing_training_repeated':False,'train_run':str(root/NAME),'child_pid':None,'phases':[],
           'source_branch':git('branch','--show-current'),'source_commit':git('rev-parse','HEAD'),
           'source_sha256':source_hashes(),'data_yaml_sha256':sha256_file(root/'data.yaml'),
           'annotations':str(args.annotations.resolve()),'annotation_sha256':sha256_file(args.annotations),
           'images':str(args.images.resolve()),'deployment_python':str(args.deployment_python.absolute()),
           'reused_checkpoints':{k:{'path':str(p),'sha256':d} for k,(p,d) in CHECKPOINTS.items()},'warnings':[]}

    def save():
        state['updated_utc']=now()
        progress(root,state)
        write_atomic(root/'status.json',state)

    def frozen():
        if (source_hashes()!=state['source_sha256'] or sha256_file(root/'data.yaml')!=state['data_yaml_sha256']
                or sha256_file(args.annotations)!=state['annotation_sha256']):
            raise RuntimeError('Frozen source or data identity changed')

    def stage(name,command,required=True):
        frozen()
        item={'name':name,'command':command,'status':'running','started_utc':now(),'log':str(root/'logs'/(name+'.log'))}
        state.update(phase=name)
        state['phases'].append(item)
        with Path(item['log']).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid
            save()
            while True:
                try:
                    code=child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    save()
        state['child_pid']=None
        item.update(returncode=code,status='completed' if code==0 else 'failed',finished_utc=now())
        save()
        if code and required:
            raise RuntimeError(f'{name} exited {code}; see {item["log"]}')
        if code:
            state['warnings'].append(f'{name} failed with exit {code}; native results are preserved')
        frozen()
        return code==0

    save()
    try:
        audit_script=ROOT/'docs/experiment_reports/2026-10-24_5090_baselines/audit_assets_5090.py'
        stage('data-audit',[sys.executable,str(audit_script),'--output',str(root/'asset_audit.json')])
        audit=read(root/'asset_audit.json'); original=read(audit_script.parent/'raw/asset_audit_5090.json')
        assert audit['status']=='passed' and audit['coco']['sha256']==original['coco']['sha256']==state['annotation_sha256']
        assert all(audit[k]==original[k] for k in ('label_content_sha256','image_inventory_sha256'))
        joint=read(JOINT/'5090-c2-la-150e-s0/experiment.json')
        assert joint['status']=='completed' and joint['completed_epochs']==150 and joint['seed']==0
        assert state['data_yaml_sha256']==joint['data_yaml_sha256']
        # Historical experiment may lack subsequently added scripts; actual old
        # source files must all retain their exact bytes for this paired run.
        assert all(state['source_sha256'].get(k)==v for k,v in joint['source_sha256'].items())
        model=build_model(str(MODELS['la']))
        initial=model.ablation_initialization
        assert initial['shared_state_sha256']==joint['initialization']['shared_state_sha256']
        assert initial['p5_slot_sha256']==joint['initialization']['p5_slot_sha256']
        assert initial['r0_full_state_sha256']==joint['initialization']['r0_full_state_sha256']
        for arm in ('c0','c2'):
            old=read(OLD/f'5090-rtdetr-r18-{arm}-150e-s0/experiment_5090.json')
            assert old['status']=='completed' and old['completed_epochs']==150 and old['seed']==0
            assert old['initialization']['r0_full_state_sha256']==initial['r0_full_state_sha256']
        plan={'variant':'la','epochs':150,'fraction':1,'seed':0,'pretrained':False,'resume':False,
              'checkpoint_initialization':None,'new_training':['la'],'initialization':initial,
              'full_initial_state_sha256':tensor_digest(model.state_dict()),'config_sha256':sha256_file(MODELS['la']),
              'source_sha256':state['source_sha256'],'reused_checkpoints':state['reused_checkpoints'],
              'matching_joint':{'shared_state_sha256':initial['shared_state_sha256'],'p5_slot_sha256':initial['p5_slot_sha256']}}
        write_atomic(root/'plan.json',plan)
        del model
        stage('train-150e',[sys.executable,'-u','-m','experiments.p2_p5_la_5090.train','--variant','la',
              '--data',str(root/'data.yaml'),'--project',str(root),'--name',NAME,'--epochs','150','--seed','0','--fraction','1'])
        trained=read(root/NAME/'experiment.json')
        assert trained['status']=='completed' and trained['completed_epochs']==150 and trained['fraction']==1
        assert trained['recipe']==joint['recipe'], 'Actual training recipe differs from the joint experiment'
        assert all(trained['initialization'][k]==initial[k] for k in ('shared_state_sha256','p5_slot_sha256','p2_slot_sha256'))
        stage('evaluate-full548',[sys.executable,'-u','-m','experiments.p2_p5_la_5090.evaluate','--variant','la',
              '--run',str(root/NAME),'--annotations',str(args.annotations.resolve()),'--images',str(args.images.resolve()),
              '--output',str(root/'evaluation')])
        # Preserve the virtualenv entry path: resolving a Python symlink would
        # bypass venv site-packages and hide the installed TensorRT module.
        deployed=str(args.deployment_python.absolute())
        good=stage('benchmark-four-trained',[deployed,'-u','-m','experiments.p2_p5_la_5090.run_la_only_150',
                   '--stage','benchmark','--root',str(root)],required=False)
        if good:
            timing=read(root/'benchmark/assessment.json')
            if timing['status']!='completed':
                state['warnings'].append('Some deployment backends failed; inspect benchmark/assessment.json')
            for pair,aliases,native,training in (
                ('c0-la',{'c2':'c0','c2-la':'la'},root/'evaluation/summary.json',root/NAME/'experiment.json'),
                ('c2-joint',{'c2':'c2','c2-la':'c2-la'},JOINT/'evaluation/summary.json',JOINT/'5090-c2-la-150e-s0/experiment.json')):
                selected={alias+'/trt_fp16':timing['conditions'][actual+'/trt_fp16'] for alias,actual in aliases.items()}
                if any(v['status']!='completed' for v in selected.values()):
                    state['warnings'].append('FP16 COCO skipped for '+pair+' because an engine failed')
                    continue
                view={'status':'completed','input':timing['input'],'conditions':selected,'actual_arm_by_alias':aliases}
                view_path=root/('fp16-benchmark-view-'+pair+'.json')
                write_atomic(view_path,view)
                stage('fp16-coco-'+pair,[deployed,'-u','-m','experiments.p2_p5_la_5090.evaluate_trt_fp16_5090',
                    '--benchmark',str(view_path),'--training',str(training),'--native-evaluation',str(native),
                    '--annotations',str(args.annotations.resolve()),'--images',str(args.images.resolve()),
                    '--output',str(root/('fp16-evaluation-'+pair))],required=False)
        report(root)
        state.update(status='completed_with_gaps' if state['warnings'] else 'completed',phase='finished',completed_utc=now())
        save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        save()
        raise


def benchmark(root):
    import fcntl
    import torch
    from experiments.decoder_budget_5090.profile_runtime_5090 import DEFAULT_DATA,configure,environment,load_input,measure,sha,stats,write
    from experiments.decoder_budget_5090.screen_decoder_perf_5090 import PredictionOnly,ready_paths
    from experiments.p2_p5_la_5090.model import inspect_model
    with open('/tmp/zhj-uav-5090-gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        configure()
        output=root/'benchmark';output.mkdir(exist_ok=False)
        ev=read(root/'evaluation/summary.json')
        weights={k:p for k,(p,_) in CHECKPOINTS.items()};weights['la']=Path(ev['checkpoint']['path'])
        assert sha(weights['la'])==ev['checkpoint']['sha256']
        for arm,(path,digest) in CHECKPOINTS.items():assert sha(path)==digest,arm
        with torch.inference_mode():
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                x,identity=load_input(DEFAULT_DATA)
                functions={};conditions={};keep=[]
                for arm in ('c0','c2','la','c2-la'):
                    path=weights[arm];saved=torch.load(path,map_location='cpu',weights_only=False)
                    model=(saved.get('ema') or saved['model']).float().eval().cuda()
                    arch=inspect_model(model,arm)
                    fns,records,objects=ready_paths(PredictionOnly(model),[x],['images'],output/arm,True,
                                                    ['eager_fp32','trt_fp32','trt_fp16'])
                    keep.extend(objects)
                    for backend,record in records.items():
                        key=arm+'/'+backend
                        record['checkpoint']={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
                        record['architecture']=arch;conditions[key]=record
                    functions.update({arm+'/'+k:v for k,v in fns.items()})
                for fn in functions.values():
                    for _ in range(150):fn()
                torch.cuda.synchronize()
                order_by_round=[]
                for index in range(20):
                    order=list(functions);random.Random(20261027+index).shuffle(order);order_by_round.append(order)
                    for key in order:
                        block=measure(functions[key],10,50);block['round']=index+1;conditions[key]['blocks'].append(block)
                    print('paired timing round',index+1,flush=True)
                for key in functions:
                    item=conditions[key]
                    for clock in ('cuda','wall'):
                        item[clock+'_stats_ms']=stats([v for b in item['blocks'] for v in b[clock+'_ms']])
                    item['status']='completed'
                write(output/'assessment.json',{'status':'completed' if len(functions)==12 else 'partial_backends_failed',
                      'environment':environment('cuda:0'),'input':identity,'conditions':conditions,'round_order':order_by_round,
                      'protocol':{'rounds':20,'samples_per_round':50,'initial_warmup':150,
                                  'scope':'GPU-resident batch1 640 pure model forward; no pre/postprocess/transfers',
                                  'weights':'own best checkpoints; existing three reused without retraining'}})
            stream.synchronize()


def report(root):
    from experiments.decoder_budget_5090.report_decoder_screen_5090 import reduction
    native={'c0':read(ROOT/'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-c0.json')['conditions']['d3']['metrics'],
            'c2':read(ROOT/'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-c2.json')['conditions']['d3']['metrics'],
            'la':read(root/'evaluation/summary.json')['metrics'],'c2-la':read(JOINT/'evaluation/summary.json')['metrics']}
    destination=root/'report';destination.mkdir(exist_ok=False)
    rows=[{'variant':arm,**native[arm]} for arm in ('c0','c2','la','c2-la')]
    with (destination/'accuracy_fp32.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    effects={}
    for key in ('AP','APs','APm','APl','AP50','AP75'):
        c0,c2,la,joint=[100*native[a][key] for a in ('c0','c2','la','c2-la')]
        effects[key]={'LA_without_P2_pp':la-c0,'LA_with_P2_pp':joint-c2,
                      'P2_without_LA_pp':c2-c0,'P2_with_LA_pp':joint-la,
                      'interaction_pp':joint-c2-la+c0}
    (destination/'effects.json').write_text(json.dumps(effects,indent=2,allow_nan=False)+'\n')
    lines=['# LA-only 150轮与已有5090模型对照','',
           '仅LA-only为本轮新训练；C0/C2/C2＋LA复用已训练seed0、同数据及配方150轮结果。', '',
           '| 模型 | 原生FP32 AP | APs | APm | APl |','|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['variant']+' | '+' | '.join(f"{100*row[k]:.3f}" for k in ('AP','APs','APm','APl'))+' |')
    lines += ['',f"AP交互项（组合−C2−LA＋C0）：{effects['AP']['interaction_pp']:+.3f}百分点；APs交互项：{effects['APs']['interaction_pp']:+.3f}百分点。",
              '这是一组seed0固定预算估计，不代表统计显著性或充分收敛。','']
    timing_path=root/'benchmark/assessment.json'
    if timing_path.exists():
        timing=read(timing_path)
        lines += ['| 后端 | C0/ms | C2/ms | LA/ms | C2＋LA/ms |','|---|---:|---:|---:|---:|']
        for backend in ('eager_fp32','trt_fp32','trt_fp16'):
            cells=[]
            for arm in ('c0','c2','la','c2-la'):
                v=timing['conditions'][arm+'/'+backend]
                cells.append(f"{v['cuda_stats_ms']['mean']:.4f}" if v['status']=='completed' else '失败')
            lines.append('| '+backend+' | '+' | '.join(cells)+' |')
        effects_timing={}
        for backend in ('eager_fp32','trt_fp32','trt_fp16'):
            for a,b in (('c0','la'),('c2','c2-la'),('c0','c2'),('la','c2-la')):
                x,y=timing['conditions'][a+'/'+backend],timing['conditions'][b+'/'+backend]
                if x['status']==y['status']=='completed':effects_timing[f'{backend}:{a}->{b}']=reduction(x,y)
        (destination/'timing_effects.json').write_text(json.dumps(effects_timing,indent=2,allow_nan=False)+'\n')
        lines += ['', '20轮×50次，GPU常驻batch1/640纯forward；配对差及95%区间见 timing_effects.json。','']
    pairs=[('c0-la',{'c2':'c0','c2-la':'la'}),('c2-joint',{'c2':'c2','c2-la':'c2-la'})]
    deployment={}
    for pair,aliases in pairs:
        path=root/('fp16-evaluation-'+pair)/'summary.json'
        if path.exists():
            d=read(path)
            for alias,actual in aliases.items():deployment[actual]=d['models'][alias]['metrics']
    if deployment:
        lines += ['| 模型 | TensorRT FP16正式AP | FP16 APs |','|---|---:|---:|']
        for arm in ('c0','c2','la','c2-la'):
            m=deployment.get(arm)
            lines.append(f"| {arm} | {100*m['AP']:.3f} | {100*m['APs']:.3f} |" if m else f'| {arm} | 未完成 | 未完成 |')
        lines += ['', '实际FP16引擎完整548图、batch1，输入输出FP32，原生FP32为batch2；两条路径分别评估。','']
    lines += ['权重、逐图预测、完整日志和导出引擎仅服务器保留。训练曲线/原始计时及身份见各自run；完成后归档文本证据。','']
    (destination/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('run','benchmark'),default='run')
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--annotations',type=Path)
    p.add_argument('--images',type=Path)
    p.add_argument('--deployment-python',type=Path)
    args=p.parse_args()
    if args.stage=='benchmark':benchmark(args.root.resolve())
    else:
        if not all((args.data,args.annotations,args.images,args.deployment_python)):
            p.error('run needs data, annotations, images and deployment-python')
        controller(args)
