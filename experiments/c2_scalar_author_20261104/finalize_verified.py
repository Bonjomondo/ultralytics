"""Verify completed scalar run and archive evaluation recovery without retraining."""
from pathlib import Path
import csv,json,shutil,subprocess
from controller import ROOT,RUN,REPORT,sha,write,now,finish


def main():
    saved=json.loads((ROOT/'controller_status.json').read_text())
    write(ROOT/'evaluation_failure_350.json',saved)
    assert len(list(csv.DictReader((RUN/'results.csv').open())))==350
    meta=json.loads((RUN/'evaluation/prediction_metadata.json').read_text())
    assert sha(RUN/'weights/best.pt')==meta['checkpoint_sha256']
    assert sha(Path(meta['gt_path']))==meta['gt_sha256']
    assert sha(RUN/'evaluation/predictions.json')==meta['prediction_sha256']
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    assert all(sha(ROOT/k)==v for k,v in manifest.items()),'Original frozen source changed'
    finish()
    d=json.loads((ROOT/'final_result.json').read_text())
    d['limitations']=[v for v in d['limitations'] if 'workers8 retained' not in v]
    d['limitations']+=['DataLoader worker SIGSEGV at epoch131; raw FP32 state restored from completed epoch130; workers8 changed to0; not bitwise equivalent continuation.',
        'First independent CPU COCOeval segfaulted at evaluateImg. Unmodified script rerun with the same GT and predictions succeeded. Native root cause remains unknown.',
        'Candidate evaluation uses full FP32 saved EMA; historical controls derive from FP16 checkpoints cast to FP32. Single-seed recovered comparison is exploratory.']
    d['recovery_receipt']=json.loads((ROOT/'worker_failure_130/receipt.json').read_text())
    d['restore_verification']=json.loads((ROOT/'recovery_verification.json').read_text())
    d['verification']={'best_gt_prediction_hashes_recomputed':True,'frozen_source_hashes_pass':True,
        'complete_csv_epochs':350,'benchmark_checkpoint_hashes_match_metrics':True,'checked_utc':now()}
    write(ROOT/'final_result.json',d);write(REPORT/'final_result.json',d)
    for name in ['recovery_receipt.json','recovery_verification.json']:
        if name=='recovery_receipt.json':shutil.copyfile(ROOT/'worker_failure_130/receipt.json',REPORT/name)
        else:shutil.copyfile(ROOT/name,REPORT/name)
    raw=REPORT/'raw/scalar';raw.mkdir(parents=True,exist_ok=True)
    for name in ['results.csv','args.yaml','initialization.json']:
        shutil.copyfile(RUN/name,raw/name)
    assert sha(raw/'results.csv')==sha(RUN/'results.csv')
    plot_records={}
    for p in [RUN/'results.png',*(RUN/'evaluation').glob('*.png')]:
        if not p.exists():continue
        dst=REPORT/'figures'/p.name;dst.parent.mkdir(exist_ok=True);shutil.copyfile(p,dst)
        plot_records[str(dst.relative_to(REPORT))]={'sha256':sha(dst),'protocol':'framework native validation curves/confusion; not independent COCO metrics'}
    write(REPORT/'figure_manifest.json',plot_records)
    trace=(ROOT/'recovery_coco.log').read_text(errors='replace')
    (REPORT/'coco_failure_trace.txt').write_text(trace)
    (REPORT/'coco_retry_evidence.txt').write_text((ROOT/'coco_retry1.log').read_text(errors='replace'))
    s=saved;s.update(status='complete',child_pid=None,finished_utc=now())
    s['stages'] += [{'phase':'coco_retry1','status':'complete','returncode':0,'log':str(ROOT/'coco_retry1.log')},
                    {'phase':'benchmark_retry1','status':'complete','returncode':0,'log':str(ROOT/'benchmark_retry1.log')},
                    {'phase':'finalize_verified','status':'complete','returncode':0}]
    write(ROOT/'controller_status.json',s);write(REPORT/'final_controller_status.json',s)
    p=REPORT/'README.md';p.write_text(p.read_text()+'\n## 最终结果\n\n350轮训练和548图预测完成；独立COCO评估第一次SIGSEGV后以同一GT/预测重跑成功，三组同场测速完成。完整结果见[RESULTS.md](RESULTS.md)与[comparison.csv](comparison.csv)。\n\nScalar AP=28.038、APs=19.840；对B0 AP+0.265pp、APs+0.359pp；对C2 AP−0.684pp、APs−0.135pp。本次没有体现新增门控相对C2的精度收益。单seed、worker恢复和checkpoint精度差异限制见final_result.json。训练预算完成，不能直接宣称充分收敛。\n\n完整350轮原始曲线见raw/scalar/results.csv；figures/为框架原生曲线，不能替代正式COCO AP。CPU评估故障栈见coco_failure_trace.txt，重跑日志见coco_retry_evidence.txt；没有重新训练或重生成预测。\n')
    print(json.dumps({'rows':d['rows'],'effects_pp':d['effects_pp'],'verification':d['verification']},ensure_ascii=False))

if __name__=='__main__':main()
