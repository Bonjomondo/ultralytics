"""Verify compact archived stage tables and hashes without server-only assets."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def rows(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))
def verify():
    for line in (ROOT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1)
        p=(ROOT/name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file(), name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, name
    metrics=rows('formal_metrics.csv'); delta=rows('paired_stage_deltas.csv'); coverage=rows('coverage.csv')
    assert len(metrics)==18 and len(delta)==9
    index={(int(r['seed']),r['model'],r['stage']):r for r in metrics}
    assert len(index)==18
    for r in delta:
        for k,v in r.items():
            if not k.startswith('delta_'): continue
            metric=k[len('delta_'):-len('_pp')]
            seed=int(r['seed']); stage=r['stage']
            assert abs(float(v)-(float(index[seed,'c2',stage][metric])-float(index[seed,'c0',stage][metric])))<1e-10
    for r in coverage:
        n=int(r['denominator'])
        if n:
            assert abs(float(r['coverage'])-int(r['covered'])/n)<1e-12
            assert abs(float(r['class_correct_coverage'])-int(r['class_correct_covered'])/n)<1e-12
            assert 0<=int(r['class_correct_covered'])<=int(r['covered'])<=n
        if r['group']=='all': assert n==38759
    cohorts=rows('advantage_cohorts.csv')
    totals={}
    for r in cohorts:
        if r['dimension']!='all': continue
        key=(r['seed'],r['iou'],r['kind'],r['transition'])
        totals[key]=totals.get(key,0)+int(r['gt_count'])
        assert abs(float(r['fraction'])-int(r['gt_count'])/int(r['from_cohort_denominator']))<1e-12
    assert len(totals)==24 and set(totals.values())=={38759}
    for r in rows('formal_gt_transitions.csv'):
        if r['group']=='all':
            assert sum(int(r[k]) for k in ('lost','gained','stable_tp','stable_fn'))==38759
    info=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    assert info['status']=='completed' and info['seeds']==[0,1,2] and info['no_training']
    assert len(info['models'])==6 and len({r['input_gt_identity_sha256'] for r in info['models']})==1
    print('PASS: hashes, 18 formal rows, 9 paired stage rows, coverage denominators, six-model input/GT identity; no GPU/server assets needed.')
if __name__=='__main__': verify()
