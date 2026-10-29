"""Offline archive integrity checks only; standard library, no model/data loading."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
counts={}
for folder in ['experiments/catapro_baseline_ramer_cluster40_20261029',
               'cache_origin/catapro_model2_ramer_cluster40_20261028']:
    base=ROOT/folder
    manifest=read(base/'deployment_manifest.json')
    for name,expected in manifest.items():
        assert digest(base/name)==expected,name
    counts[folder]=len(manifest)
baseline=ROOT/'experiments/catapro_baseline_ramer_cluster40_20261029'
identity=read(baseline/'source_identity.json')
for name,record in identity.items():
    assert digest(baseline/name)==record['deployed_sha256'],name
    if name!='runtime/master_store.py':
        assert record['deployed_sha256']==record['original_sha256'],name
raw=(baseline/'runtime/master_store.py').read_bytes()
assert raw.count(b"{'name': 'RAMER', 'width': 1024}")==1
restored=raw.replace(b"{'name': 'RAMER', 'width': 1024}",b"{'name': 'ProtT5', 'width': 1024}")
assert hashlib.sha256(restored).hexdigest()==identity['runtime/master_store.py']['original_sha256']
python_files=list(ROOT.rglob('*.py'))
for p in python_files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
for p in ROOT.rglob('*.sbatch'):
    data=p.read_bytes();assert data.startswith(b'#!/bin/bash\n') and b'\r' not in data,str(p)
summary=read(ROOT/'evidence/run_seed668/summary.json')
reference=read(baseline/'reference_protocol.json')
assert summary['parameters']==reference['parameters']
assert summary['source_sha256']==reference['source_sha256']
assert summary['best_epoch_zero_based']==7 and summary['epochs_ran']==28
assert read(ROOT/'evidence/result_audit.json')['status']=='passed'
print(json.dumps(dict(status='passed',frozen_files=counts,python_files=len(python_files),
    sole_reader_change_verified=True,protocol_unchanged=True,scope='archive integrity, syntax and recorded evidence; no GPU or new numerical verification'),indent=2))
