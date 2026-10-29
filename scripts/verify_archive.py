"""Offline archive integrity checks only; standard library, no model/data loading."""
import ast
import csv
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
data_root=ROOT/'datasets/cluster40_seed668'
data_manifest=read(data_root/'manifest.json')
assert data_manifest['rowspec_sha256']==digest(baseline/'inputs/rowspec.json')
assert data_manifest['assignments_sha256']==digest(baseline/'assignments.csv')
spec=read(baseline/'inputs/rowspec.json')
assignments=list(csv.DictReader((baseline/'assignments.csv').open(encoding='utf-8',newline='')))
membership={r['balance_row_uid'].removeprefix('catapro:'):r for r in assignments}
seen=set();groups={}
for split,entry in data_manifest['files'].items():
    path=data_root/entry['file'];assert digest(path)==entry['sha256']
    rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    ids=[r['author_row_id'] for r in rows]
    pairs=spec['eval']['test'] if split=='test' else spec[split]
    assert ids==[uid for _,uid in pairs] and len(ids)==entry['rows']
    assert not seen.intersection(ids) and len(ids)==len(set(ids));seen.update(ids)
    groups[split]={r['cluster_id'] for r in rows}
    assert len(groups[split])==entry['clusters']
    for row in rows:
        assignment=membership[row['author_row_id']]
        assert row['split']==assignment['new_split']==split
        assert row['cluster_id']==assignment['balance_group']
assert len(seen)==data_manifest['rows']==27658
for a,b in [('train','valid'),('train','test'),('valid','test')]:assert not groups[a]&groups[b]
print(json.dumps(dict(status='passed',frozen_files=counts,python_files=len(python_files),
    sole_reader_change_verified=True,protocol_unchanged=True,dataset_rows=len(seen),dataset_sha_and_split_verified=True,
    scope='archive integrity, syntax and recorded evidence; no GPU or new numerical verification'),indent=2))
