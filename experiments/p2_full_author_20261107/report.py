"""Archive compact evidence and rebuild the comparison from tracked JSON."""
import csv
import json
import shutil
from pathlib import Path

from common import ARMS, REPORT, RUNS, now, sha, write

PLOTS = {'results.png','F1_curve.png','PR_curve.png','P_curve.png','R_curve.png',
         'confusion_matrix.png','confusion_matrix_normalized.png'}


def archive(name):
    run = RUNS / name
    dst = REPORT / 'raw' / name
    dst.mkdir(parents=True, exist_ok=True)
    for filename in ['results.csv','args.yaml','initialization.json','first_batch.json','status.json']:
        if (run/filename).exists():
            shutil.copyfile(run/filename, dst/filename)
    for path in run.glob('restore_epoch*.json'):
        shutil.copyfile(path, dst/path.name)
    if (run/'diagnostics.jsonl').exists():
        rows = []
        for line in (run/'diagnostics.jsonl').read_text().splitlines():
            item = json.loads(line)
            counts = item['query_counts_first3_batches']
            total = [sum(x[i] for x in counts) for i in range(len(item['strides']))]
            row = {'epoch':item['epoch'], 'epoch_seconds':item['epoch_seconds'],
                   'peak_gpu_gib':item['peak_gpu_gib'],
                   'p2_grad_mean':sum(item['p2_gradient_norm_first3_batches'])/max(1,len(item['p2_gradient_norm_first3_batches']))}
            row.update({f'query_stride{s}_fraction':n/max(1,sum(total)) for s,n in zip(item['strides'],total)})
            rows.append(row)
        if rows:
            with (dst/'diagnostics.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
    for directory in sorted((run/'evaluation').glob('e*')):
        target = dst/'evaluation'/directory.name
        target.mkdir(parents=True,exist_ok=True)
        for filename in ['metrics.json','prediction_metadata.json']:
            if (directory/filename).exists():shutil.copyfile(directory/filename,target/filename)
    for directory in [run,run/'evaluation/e350_best']:
        for path in directory.glob('*.png'):
            if path.name in PLOTS:
                target = REPORT/'figures'/name/directory.name/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
    write(dst/'server_evidence.json',{'run':str(run),
          'full_diagnostics_sha256':sha(run/'diagnostics.jsonl') if (run/'diagnostics.jsonl').exists() else None,
          'sampling':'query origins and pre-clipping P2 gradient, first three batches/epoch',
          'native_figure_protocol':'author internal validation; independent COCO metrics in evaluation JSON'})
    if (run/'results.csv').exists():assert sha(run/'results.csv')==sha(dst/'results.csv')


def analyze():
    rows = []
    benchmark = json.loads((REPORT/"benchmark_final.json").read_text())["models"] if (REPORT/"benchmark_final.json").exists() else {}
    for path in sorted((REPORT/'raw').glob('main_*/evaluation/*/metrics.json')):
        meta = json.loads(path.read_text())
        name = path.parents[2].name
        arm = name.split('_')[1]
        performance = benchmark.get(name) if meta['budget'] == 350 and meta['kind'] == 'best' else None
        if performance:
            assert performance['checkpoint_sha256'] == meta['checkpoint_sha256']
        rows.append({'run':name,'arm':arm,'budget':meta['budget'],'kind':meta['kind'],
                     'checkpoint_epoch':meta['checkpoint_epoch'],**meta['coco_percent'],
                     'checkpoint_sha256':meta['checkpoint_sha256']})
        if performance:
            rows[-1].update({key:performance[key] for key in ['parameters','thop_gflops_unfused','median_ms','p95_ms']})
            rows[-1]['forward_fps'] = 1000/performance['median_ms']
    for row in rows:
        baseline = next((x for x in rows if x['arm']=='b0' and x['budget']==row['budget'] and x['kind']==row['kind']),None)
        if baseline:
            row['delta_AP_vs_b0_pp']=row['AP']-baseline['AP']
            row['delta_APs_vs_b0_pp']=row['APs']-baseline['APs']
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (REPORT/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# P2完整四尺度实验结果','','当前已归档数据如下。正式指标为converted VisDrone val548、COCO maxDets100。',
           '','| arm | budget | checkpoint | AP | APs | AP75 | ΔAP vs B0 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['arm']} | {row['budget']} | {row['kind']}@{row['checkpoint_epoch']} | {row['AP']:.3f} | {row['APs']:.3f} | {row['AP75']:.3f} | {row.get('delta_AP_vs_b0_pp',0):+.3f} |")
    lines+=['','本队列为seed1单seed实验。三种架构共有骨干及语义兼容层配对，四尺度decoder形状变更层有单独初始化记录。',
            '当轮last和截至该预算原生fitness选出的best分开评估，原生mAP不代替正式COCO AP。',
            'Tiny recall采用640方形缩放后短边分桶，COCO按类别匹配、忽略GT排除，query来源层属于初始化来源诊断。']
    if benchmark:
        lines += ['', '| arm | parameters | THOP GFLOPs | median ms | p95 ms |',
                  '|---|---:|---:|---:|---:|']
        for name, item in benchmark.items():
            lines.append(f"| {name.split('_')[1]} | {item['parameters']} | {item['thop_gflops_unfused']:.3f} | {item['median_ms']:.3f} | {item['p95_ms']:.3f} |")
    for arm in ARMS:
        name = f'main_{arm}_350e_s1'
        links = []
        candidates = [('完整CSV',f'raw/{name}/results.csv'),
                      ('诊断摘要',f'raw/{name}/diagnostics.csv'),
                      ('训练曲线',f'figures/{name}/{name}/results.png'),
                      ('PR曲线',f'figures/{name}/e350_best/PR_curve.png'),
                      ('P曲线',f'figures/{name}/e350_best/P_curve.png'),
                      ('R曲线',f'figures/{name}/e350_best/R_curve.png'),
                      ('F1曲线',f'figures/{name}/e350_best/F1_curve.png'),
                      ('混淆矩阵',f'figures/{name}/e350_best/confusion_matrix.png'),
                      ('归一化矩阵',f'figures/{name}/e350_best/confusion_matrix_normalized.png')]
        for label, target in candidates:
            if (REPORT/target).exists():links.append(f'[{label}]({target})')
        if links:lines.extend(['', f"{arm} " + '、'.join(links)])
    (REPORT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return rows


def manifest():
    files=sorted(p for p in REPORT.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256')
    (REPORT/'MANIFEST.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPORT)}\n' for p in files))


def final_audit():
    from common import GT, MILESTONES, SEED
    gt_sha = sha(GT)
    records = {}
    backbones = []
    first_batches = []
    benchmark = json.loads((REPORT/'benchmark_final.json').read_text())
    for arm in ARMS:
        name = f'main_{arm}_350e_s{SEED}'
        run = RUNS/name
        assert len(list(csv.DictReader((run/'results.csv').open()))) == 350
        assert sha(run/'results.csv') == sha(REPORT/'raw'/name/'results.csv')
        backbones.append(json.loads((run/'initialization.json').read_text())['backbone_sha256'])
        first = json.loads((run/'first_batch.json').read_text())
        first_batches.append({key:first[key] for key in ['images','image_sha256','targets_sha256']})
        evaluations = []
        for budget in MILESTONES:
            for kind in ['last','best']:
                checkpoint = run/f'milestones/e{budget}/{kind}.pt'
                output = run/f'evaluation/e{budget}_{kind}'
                metrics = json.loads((output/'metrics.json').read_text())
                assert sha(checkpoint) == metrics['checkpoint_sha256']
                assert sha(output/'predictions.json') == metrics['predictions_sha256']
                assert gt_sha == metrics['gt_sha256']
                assert metrics['images'] == 548 and metrics['detections'] == 164400
                assert metrics['checkpoint_epoch'] <= budget
                evaluations.append({'budget':budget,'kind':kind,
                    'checkpoint_epoch':metrics['checkpoint_epoch'],
                    'checkpoint_sha256':metrics['checkpoint_sha256'],
                    'predictions_sha256':metrics['predictions_sha256']})
        assert sha(run/'weights/best.pt') == benchmark['models'][name]['checkpoint_sha256']
        assert len(benchmark['models'][name]['samples_ms']) == 500
        records[arm] = {'completed_epochs':350,'curve_sha256':sha(run/'results.csv'),
                        'evaluations':evaluations,
                        'resumes':[str(p) for p in run.glob('restore_epoch*.json')]}
    assert len(set(backbones)) == 1
    assert first_batches[0] == first_batches[1] == first_batches[2]
    write(REPORT/'final_audit.json',{'status':'passed','audited_utc':now(),
          'gt_sha256':gt_sha,'benchmark_sha256':sha(REPORT/'benchmark_final.json'),
          'shared_backbone_sha256':backbones[0],'first_batch_images_targets_match':True,
          'arms':records,'limits':'seed1 only; any raw-state resumes listed per arm; no multi-seed conclusion'})
