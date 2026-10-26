"""Verify the real joint150 launch using only tracked source and JSON snapshots."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / 'raw' / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for line in (HERE / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, name
    plan, status, training = [read(n) for n in ('plan.json', 'status.json', 'training_status.json')]
    assert plan['variant'] == training['variant'] == 'c2-la' and plan['epochs'] == training['requested_epochs'] == 150
    assert plan['fraction'] == training['fraction'] == 1 and plan['seed'] == training['seed'] == 0
    assert plan['pretrained'] is False and plan['resume'] is False and plan['checkpoint_initialization'] is None
    assert status['new_training'] == ['c2-la'] and status['retrain_existing'] is False
    assert status['speed_screen_blocks_training'] is False
    assert training['status'] == 'running' and training['role'] == 'fixed_budget_screen'
    assert training['completed_epochs'] >= 1
    health = read('first_epoch_health.json')
    assert health['epoch'] == 1
    assert all(health[k] > 0 for k in ('p2_first_weight_grad_abs', 'p5_qkv_grad_abs', 'p5_rpb_grad_abs'))
    assert all(v == 1 for k, v in health.items() if k.endswith('_finite'))
    assert training['architecture']['p2_injection'] and training['architecture']['p5_na7_rpb']
    assert training['architecture']['queries'] == 300 and training['architecture']['decoder_layers'] == 3
    assert training['initialization']['shared_state_verified']
    for key in ('shared_state_sha256', 'p2_slot_sha256', 'p5_slot_sha256'):
        assert plan['initialization'][key] == training['initialization'][key]
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert status['source_sha256'] == training['source_sha256'] == plan['source_sha256']
    assert sha(HERE / 'raw/data.yaml') == training['data_yaml_sha256'] == status['data_yaml_sha256']
    audit = read('asset_audit.json')
    assert audit['status'] == 'passed' and audit['splits']['train']['images'] == 6471 and audit['splits']['val']['images'] == 548
    assert audit['coco']['sha256'] == status['annotation_sha256']
    assert status['reused_baseline'] == plan['reused_baseline']
    assert status['reused_baseline']['sha256'] == 'ac2b02c45596bb816c4a42799d2b97ff02d62a4dc153bc5816502f02e936a5d6'
    for target in re.findall(r'\]\(([^)]+)\)', (HERE / 'README.md').read_text()):
        if not target.startswith(('http:', 'https:')):
            assert (HERE / target.split('#', 1)[0]).exists(), target
    print('Verified actual fresh full150 joint launch, paired initialization, source/data identity and read-only baseline reuse.')


if __name__ == '__main__':
    main()
