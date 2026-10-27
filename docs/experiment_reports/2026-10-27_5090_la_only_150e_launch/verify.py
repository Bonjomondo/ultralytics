"""Verify the real LA-only full150 launch using tracked text, no GPU/data."""
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def read(name):
    return json.loads((HERE/'raw'/name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for line in (HERE/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);path=(ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    plan,status,training=[read(n) for n in ('plan.json','status.json','training_status.json')]
    assert plan['variant']==training['variant']=='la' and plan['epochs']==training['requested_epochs']==150
    assert plan['seed']==training['seed']==0 and plan['fraction']==training['fraction']==1
    assert plan['pretrained'] is False and plan['resume'] is False and plan['checkpoint_initialization'] is None
    assert status['new_training']==['la'] and status['existing_training_repeated'] is False
    assert training['status']=='running' and training['role']=='fixed_budget_screen' and training['completed_epochs']>=1
    arch=training['architecture']
    assert not arch['p2_injection'] and arch['p5_na7_rpb'] and arch['queries']==300 and arch['decoder_layers']==3 and arch['points_per_level']==4
    joint=json.loads((ROOT/'docs/experiment_reports/2026-10-27_5090_c2_la_150e_results/raw/training.json').read_text())
    for key in ('shared_state_sha256','p5_slot_sha256','r0_full_state_sha256'):
        assert plan['initialization'][key]==training['initialization'][key]==joint['initialization'][key]
    preflight=json.loads((ROOT/'docs/experiment_reports/2026-10-26_5090_p2_p5_la_preflight/raw/benchmark.json').read_text())
    assert plan['initialization']['p2_slot_sha256']==preflight['models']['c0']['identity']['initialization']['p2_slot_sha256']
    assert all(plan['source_sha256'].get(k)==v for k,v in joint['source_sha256'].items())
    assert status['source_sha256']==training['source_sha256']==plan['source_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(HERE/'raw/data.yaml')==training['data_yaml_sha256']==joint['data_yaml_sha256']
    audit=read('asset_audit.json')
    assert audit['status']=='passed' and audit['splits']['train']['images']==6471 and audit['splits']['val']['images']==548
    assert audit['coco']['sha256']==status['annotation_sha256']
    expected={'c0':'adeaec17109a3b3ac6b8a6e521fb6cc3c2bdb3cd124ad6eedecc930d35a03d2e',
              'c2':'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6',
              'c2-la':'1a3866ebc5246748d8c359acb30129baa688566c41cfe3be5ce6a05baab4833f'}
    assert {k:v['sha256'] for k,v in status['reused_checkpoints'].items()}==expected
    assert status['reused_checkpoints']==plan['reused_checkpoints']
    for arm,ap in (('c0',.18661578295635495),('c2',.19516708680487616)):
        old=json.loads((ROOT/f'docs/experiment_reports/2026-10-25_5090_decoder_screen/raw/depth-eval-{arm}.json').read_text())
        assert old['checkpoint']['sha256']==expected[arm] and abs(old['conditions']['d3']['metrics']['AP']-ap)<1e-12
    joint_eval=json.loads((ROOT/'docs/experiment_reports/2026-10-27_5090_c2_la_150e_results/raw/evaluation.json').read_text())
    assert joint_eval['checkpoint']['sha256']==expected['c2-la'] and abs(joint_eval['metrics']['AP']-.19026708410453486)<1e-12
    h=read('first_epoch_health.json')
    assert h['epoch']==1 and h['p5_qkv_grad_abs']>0 and h['p5_rpb_grad_abs']>0
    assert 'p2_first_weight_grad_abs' not in h
    assert all(v==1 for k,v in h.items() if k.endswith('_finite'))
    for target in re.findall(r'\]\(([^)]+)\)',(HERE/'README.md').read_text()):
        if not target.startswith(('http:','https:')):assert (HERE/target.split('#',1)[0]).exists(),target
    print('Verified real LA-only 150e launch, no P2, matched P5/common initialization, source/data identity and first-epoch gradients.')


if __name__=='__main__':main()
