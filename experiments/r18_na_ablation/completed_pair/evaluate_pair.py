"""Evaluate completed R0 vs R1 with identical native JSON / COCOeval protocol.

Existing R1 native predictions are reused only after checkpoint/source/result
identity validation. This supplements the earlier R1 trajectory validation.
"""
from collections import Counter
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
import yaml
from ultralytics import RTDETR
from detrscope.core.io import read_json,write_json,sha256_file
from detrscope.evaluation.coco import CocoEvaluator,compare_gt

p=Path(__file__).resolve().parent;repo=p.parents[2]
old=p.parent/'detrscope_20261021'
runs=repo/'runs/detect/r18_na_ablation'
gt=Path('/home/zjnu/文档/ZHJ_WorkSpace/datasets/VisDrone2019-DET-YOLO/annotations/VisDrone2019-DET_val_coco.json')
parser=argparse.ArgumentParser()
parser.add_argument('--raw',type=Path,default=runs/'completed_pair_evaluation')
raw=parser.parse_args().raw.resolve();raw.mkdir(parents=True,exist_ok=False)
write_json(raw/'status.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()})
try:
    torch.set_num_threads(4)
    pin=read_json(old/'source_pin.json')
    assert {str(f.relative_to(repo)):sha256_file(f) for f in sorted((repo/'ultralytics').rglob('*.py'))}==pin['source_sha256']
    dataset=read_json(gt);images={Path(im['file_name']).name:im['id'] for im in dataset['images']}
    evaluator=CocoEvaluator(gt);evaluated={};identity={};logs={}
    args={};init={}
    for variant in ('r0','r1'):
        run=runs/f'rtdetr-r18-{variant}-150e-s0';weight=run/'weights/best.pt'
        with (run/'results.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==150 and int(rows[-1]['epoch'])==150
        logs[variant]={'final':rows[-1],'best':max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))}
        args[variant]=yaml.safe_load((run/'args.yaml').read_text())
        initialization=run/('initialization-before-resume-e36.json' if variant=='r1' else 'initialization.json')
        init[variant]=read_json(initialization)
        identity[variant]={'checkpoint':str(weight),'sha256':sha256_file(weight),'training_rows':len(rows),
            'last_checkpoint_sha256':sha256_file(run/'weights/last.pt'),'initialization':init[variant]}
        if variant=='r0':
            print('BEGIN R0 NATIVE VALIDATION',flush=True)
            model=RTDETR(str(weight))
            result=model.val(data=str(repo/'configs/visdrone-local.yaml'),device='0',imgsz=640,batch=2,workers=0,
                half=False,plots=False,verbose=False,save_json=True,project=str(runs),name='completed_pair_r0_native',exist_ok=False)
            native={'metrics':result.results_dict,'checkpoint_sha256':identity[variant]['sha256'],'save_dir':str(result.save_dir),
                    'batch':2,'imgsz':640,'half':False}
            write_json(p/'r0_native_internal.json',native)
            for src,dst in [('results.csv','r0_training_metrics.csv'),('health.jsonl','r0_training_health.jsonl'),('args.yaml','r0_training_args.yaml')]:
                shutil.copy2(run/src,p/dst)
        else:
            native=read_json(old/'native_val_internal_metrics.json')
            assert native['checkpoint_sha256']==identity[variant]['sha256']
        pred=Path(native['save_dir'])/'predictions.json';predictions=read_json(pred)
        for row in predictions:row['image_id']=images[row['file_name']]
        assert {r['image_id'] for r in predictions}==set(images.values())
        formal=json.loads(json.dumps(evaluator.evaluate(predictions),default=lambda x:x.item() if isinstance(x,np.generic) else x))
        if variant=='r1':
            prior=read_json(old/'native_val_formal_coco.json')
            assert formal['predictions_sha256']==prior['predictions_sha256']
            assert formal['metrics']==prior['metrics']
        identity[variant].update(predictions_path=str(pred),predictions_file_sha256=sha256_file(pred),native_internal=native['metrics'])
        evaluated[variant]=formal;write_json(raw/f'{variant}_coco.json',formal)
        write_json(p/f'{variant}_formal_metrics.json',{k:v for k,v in formal.items() if k!='gt_states'})
        print('FORMAL',variant,formal['metrics'],flush=True)
    assert init['r0']['shared_state_sha256']==init['r1']['shared_state_sha256']
    recipe_fields=('epochs','batch','imgsz','seed','amp','optimizer','lr0','lrf','weight_decay','nbs','warmup_epochs',
                   'warmup_bias_lr','mosaic','mixup','close_mosaic','pretrained','data','deterministic')
    differences={k:[args['r0'].get(k),args['r1'].get(k)] for k in recipe_fields if args['r0'].get(k)!=args['r1'].get(k)}
    assert not differences,differences
    changes=compare_gt(evaluated['r0'],evaluated['r1']);write_json(raw/'gt_changes.json',changes)
    compact=[{'iou':r['iou'],**r['counts']} for r in changes['thresholds']]
    write_json(p/'formal_gt_changes_summary.json',{'comparison':'R1 relative to R0','thresholds':compact})
    rows=[{'metric':k,'R0':evaluated['r0']['metrics'][k],'R1':v,'R1_minus_R0_pp':(v-evaluated['r0']['metrics'][k])*100}
          for k,v in evaluated['r1']['metrics'].items()]
    with (p/'formal_metrics_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(p/'verification.json',{'status':'completed','created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.executable,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'cpu_affinity':sorted(__import__('os').sched_getaffinity(0)),
        'source_commit':pin['commit'],'model_source_sha256_unchanged':True,'checkpoints':identity,'training_log_metrics':logs,
        'training_recipe_fields_identical':recipe_fields,'training_recipe_differences':differences,
        'shared_initialization_sha256':init['r0']['shared_state_sha256'],'protocol_identical':evaluated['r0']['protocol']==evaluated['r1']['protocol'],
        'data_images':548,'data_annotations':38759,'raw_output':str(raw),
        'scope':'Same-seed 150-epoch completed checkpoints, native fused FP32 Validator JSON (3-decimal boxes/5-decimal scores), standard COCO maxDets=[1,10,100]. R1 cached predictions/result reproduced exactly. No new training or R0 decoder trajectory collection.'})
    metric_table='\n'.join(f"| {r['metric']} | {r['R0']*100:.3f} | {r['R1']*100:.3f} | {r['R1_minus_R0_pp']:+.3f} |" for r in rows if r['metric'] in ('AP','AP50','AP75','APs','APm','APl','AR100'))
    gt_table='\n'.join(f"| {r['iou']:.2f} | {r['lost']} | {r['gained']} | {r['stable_tp']} | {r['stable_fn']} |" for r in compact if abs(r['iou']-.5)<1e-8 or abs(r['iou']-.75)<1e-8)
    result=f'''# R0 / R1 完成版对照

两组均已完成 150 轮，seed=0，完整 VisDrone 验证集 548 图 / 38,759 GT。R0 的最后 checkpoint 保存时间为主机本地 2026-10-21 18:53:59；训练日志正常结束，非中断结果。

## 当前结论

训练 CSV 内部 mAP50-95：R0 **20.593%**，R1 **20.695%**，R1−R0 **+0.102 个百分点**；mAP50 分别为 **34.982% / 35.170%**，差 **+0.188 个百分点**。这不是正式 COCOeval 数值。

下表使用相同原生 Validator 导出及标准 COCOeval 协议。R0 新增原生验证；R1 复用先前原生预测 JSON，但重新核验 checkpoint、源码 SHA、预测身份并重算，正式指标与历史结果逐值一致。R1 历史报告 18.702% 使用未融合的轨迹导出，本次使用与 R0 相同的原生融合导出，所以取其 18.705% 作为对照，避免混用路径。

| 正式指标 | R0（%） | R1（%） | R1−R0（百分点） |
| --- | ---: | ---: | ---: |
{metric_table}

本次正式 AP 差 {(evaluated['r1']['metrics']['AP']-evaluated['r0']['metrics']['AP'])*100:+.3f} 个百分点，总体差异很小；APs 差 {(evaluated['r1']['metrics']['APs']-evaluated['r0']['metrics']['APs'])*100:+.3f}，但 APm 差 {(evaluated['r1']['metrics']['APm']-evaluated['r0']['metrics']['APm'])*100:+.3f}、ARs 差 {(evaluated['r1']['metrics']['ARs']-evaluated['r0']['metrics']['ARs'])*100:+.3f} 个百分点，表现存在群体和指标上的取舍，不能概括为全面改善。

共有初始化 533 项 SHA 一致；两组关键训练配方一致，R1 曾从第 36 轮 checkpoint 续跑。以上仅是单 seed 完成版结果，不能据此宣称稳定收益或训练因果。此前“R0 尚未完成”的记录应按历史快照阅读；最新状态以此目录为准。

## 正式同 GT 匹配变化

| IoU | Lost（R0 TP→R1 FN） | Gained | Stable TP | Stable FN |
| --- | ---: | ---: | ---: | ---: |
{gt_table}

数据来自同一份 COCO GT、相同评估协议和注释 ID；不是 Hungarian 诊断责任。完整 GT ID 列表保留在 `{raw}/gt_changes.json`，未提交大体积原始预测。

## 证据与复现

- [正式对照表](formal_metrics_comparison.csv)、[R0 正式结果](r0_formal_metrics.json)、[R1 正式结果](r1_formal_metrics.json)、[身份与训练配方核验](verification.json)。
- [R0 训练曲线](r0_training_metrics.csv)、[R0 健康日志](r0_training_health.jsonl)、[R0 原生内部指标](r0_native_internal.json)。
- [R1 训练曲线](../detrscope_20261021/training_metrics.csv)、[R1 既有逐层诊断](../detrscope_20261021/REPORT.md)。本次未新增 R0 Decoder 轨迹采集，不声称已完成两组完整 Debug 机制比较。

在仓库根目录使用 UAV_CV，传入新的 `--raw` 输出目录，避免覆盖证据，再通过 tmux 启动。脚本将刷新本目录的生成表，复现后先检查 Git diff：

```bash
cd /home/zjnu/文档/ZHJ_WorkSpace/ultralytics
/home/zjnu/anaconda3/bin/tmux new-session -d -s r18-pair-eval \\
  'PYTHONPATH=$PWD:/home/zjnu/文档/ZHJ_WorkSpace/detrscope/src PYTHONMALLOC=debug CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 24-27 /home/zjnu/anaconda3/envs/UAV_CV/bin/python -X dev -u experiments/r18_na_ablation/completed_pair/evaluate_pair.py --raw "$PWD/runs/detect/r18_na_ablation/completed_pair_reproduction" >> logs/r18_completed_pair_eval.log 2>&1'
```

环境为已有 UAV_CV / PyTorch 2.1.2+cu121；SDK v0.2 源码运行的环境限制与此前报告相同。没有新增训练，没有覆盖旧 R1 验收产物。网页端 ChatGPT 分析时优先读取本报告、对照 CSV 和 verification.json，再按需查看旧逐层诊断；请区分内部指标与正式 COCOeval。
'''
    (p/'REPORT.md').write_text(result)
    for f in p.glob('*.csv'):f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
    write_json(raw/'status.json',{'status':'completed','finished_utc':datetime.now(timezone.utc).isoformat()})
    write_json(p/'artifact_index.json',{'files':{f.name:sha256_file(f) for f in p.iterdir() if f.is_file() and f.name!='artifact_index.json'}})
    print('PAIR COMPLETE',flush=True)
except BaseException as e:
    write_json(raw/'status.json',{'status':'failed','error':f'{type(e).__name__}: {e}'})
    raise
