"""Strict no-condition feature/label store for the author CataPro table.

No generation, normalization of SMILES, filtering or label repair occurs here.
All source rows and folds must already have passed the external data audit.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from residue_cache import MODEL_SHA
from support import sha

WINDOW_PREPROCESSING = 'UZOB->X; full length; BOS/EOS excluded; window1022 stride894 overlap128'
WINDOW_MERGE = 'arithmetic mean at overlapping residues, float32; CNN sees complete reconstructed sequence'


def normalize_sequence(raw):
    if not isinstance(raw, str) or not re.fullmatch('[A-Z]+', raw):
        raise RuntimeError('Missing/invalid sequence')
    return re.sub('[UZOB]', 'X', raw)


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


class MasterStore:
    def __init__(self, path):
        root = Path(path).resolve().parent
        self.manifest = json.loads(Path(path).read_text(encoding='utf-8'))
        if self.manifest.get('schema') != 'catapro_nocond_features_v1':
            raise RuntimeError('Wrong no-condition feature schema')
        paths = {}
        for key in ('rows_csv', 'features_npy'):
            paths[key] = resolve(root, self.manifest[key])
            if sha(paths[key]) != self.manifest[key + '_sha256']:
                raise RuntimeError('Feature-store hash mismatch: ' + key)
        frame = pd.read_csv(paths['rows_csv'], dtype={'author_row_id': str},
                            keep_default_na=False, float_precision='round_trip')
        required = {'author_row_id', 'fold', 'sequence', 'smiles',
                    'kcat_s_minus1', 'log10_kcat'}
        if not required.issubset(frame.columns):
            raise RuntimeError('Missing required author row metadata')
        if frame.author_row_id.duplicated().any() or (frame.author_row_id == '').any():
            raise RuntimeError('Missing/duplicate author row ID')
        if not frame.fold.isin(range(10)).all():
            raise RuntimeError('Invalid author fold')
        if (frame.sequence == '').any() or (frame.smiles == '').any():
            raise RuntimeError('Missing sequence/SMILES')
        raw = frame.kcat_s_minus1.to_numpy(dtype=np.float64)
        labels = frame.log10_kcat.to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all() or not (raw > 0).all():
            raise RuntimeError('kcat must be finite and positive in s^-1')
        if not np.isfinite(labels).all() or not np.allclose(
                labels, np.log10(raw), atol=1e-12, rtol=0):
            raise RuntimeError('log10 label audit failed: expected log10(raw kcat) exactly once')
        if self.manifest.get('feature_blocks') != [
                {'name': 'RAMER', 'width': 1024},
                {'name': 'MolT5', 'width': 768},
                {'name': 'MACCS', 'width': 167}]:
            raise RuntimeError('Feature order/dimensions are not approved')
        self.features = np.load(paths['features_npy'], allow_pickle=False, mmap_mode='r')
        if self.features.shape != (len(frame), 1959) or self.features.dtype != np.float32:
            raise RuntimeError('Expected float32 feature1959.npy aligned to rows.csv')
        if not np.isfinite(self.features).all():
            raise RuntimeError('Non-finite pooled feature')
        self.frame = frame
        self.indices = {sid: i for i, sid in enumerate(frame.author_row_id)}

    def get(self, split, sid):
        if split != 'catapro':
            raise KeyError('Expected source split catapro')
        i = self.indices[str(sid)]
        row = self.frame.iloc[i]
        return dict(features=np.array(self.features[i], copy=True),
                    label=float(row.log10_kcat), sequence=row.sequence,
                    smiles=row.smiles, fold=int(row.fold))

    def __len__(self):
        return len(self.frame)


class ManifestResidueResolver:
    """Verify each immutable ESM file once in preflight, never alter it."""
    def __init__(self, manifest_path, path_map=None):
        root = Path(manifest_path).resolve().parent
        meta = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        remap = json.loads(Path(path_map).read_text(encoding='utf-8')) if path_map else {}
        if 'file_paths' in meta:
            order_path = root / 'order.json'
            if sha(order_path) != meta['order_sha256']:
                raise RuntimeError('Existing residue order hash mismatch')
            order = json.loads(order_path.read_text(encoding='utf-8'))
            if order['preprocessing'] != WINDOW_PREPROCESSING or order['long_sequence_merge'] != WINDOW_MERGE:
                raise RuntimeError('Existing residue window policy differs')
            if meta['long_sequence_policy'] != WINDOW_MERGE:
                raise RuntimeError('Manifest long sequence policy differs')
            for obj in (order, meta):
                if obj['model_sha256'] != MODEL_SHA or obj['layer'] != 36 or obj['dimension'] != 2560:
                    raise RuntimeError('Existing residue encoder protocol differs')
            if meta['status'] != 'complete' or len(order['sequences']) != meta['unique_sequences']:
                raise RuntimeError('Incomplete existing residue cache')
            keys = {str(i) for i in range(len(order['sequences']))}
            if set(meta['file_paths']) != keys or set(meta['file_sha256']) != keys:
                raise RuntimeError('Incomplete existing residue file records')
            meta = dict(schema='catapro_nocond_residues_v1', status='complete',
                        model_sha256=MODEL_SHA, layer=36, dimension=2560,
                        dtype='float16_short_float32_long', preprocessing=WINDOW_PREPROCESSING,
                        long_sequence_merge=WINDOW_MERGE,
                        sequences={hashlib.sha256(s.encode()).hexdigest(): dict(
                            sequence=s, length=len(s), path=meta['file_paths'][str(i)],
                            sha256=meta['file_sha256'][str(i)]) for i, s in enumerate(order['sequences'])})
            if len(meta['sequences']) != len(order['sequences']):
                raise RuntimeError('Duplicate normalized cache sequence')
        expected = dict(schema='catapro_nocond_residues_v1', status='complete',
                        model_sha256=MODEL_SHA, layer=36, dimension=2560,
                        dtype='float16_short_float32_long', preprocessing=WINDOW_PREPROCESSING,
                        long_sequence_merge=WINDOW_MERGE)
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError('Wrong residue manifest: ' + key)
        self.table = {}
        for key, record in meta['sequences'].items():
            seq = normalize_sequence(record['sequence'])
            if seq != record['sequence'] or hashlib.sha256(seq.encode()).hexdigest() != key:
                raise RuntimeError('Residue manifest sequence hash mismatch')
            path = resolve(root, remap.get(record['path'], record['path']))
            if record['length'] != len(seq) or sha(path) != record['sha256']:
                raise RuntimeError('Residue length/file hash mismatch')
            values = np.load(path, allow_pickle=False, mmap_mode='r')
            dtype = np.float16 if len(seq) <= 1022 else np.float32
            if values.shape != (len(seq), 2560) or values.dtype != dtype:
                raise RuntimeError('Residue array shape/dtype mismatch')
            if not np.isfinite(values).all():
                raise RuntimeError('Non-finite residue array')
            self.table[seq] = str(path)

    def path(self, sequence):
        return self.table[normalize_sequence(sequence)]


def validate_rowspec(spec, store, resolver):
    groups = {'train': spec['train'], 'valid': spec['valid'], **spec['eval']}
    if set(spec['eval']) != {'test'} or set(groups) != {'train', 'valid', 'test'}:
        raise RuntimeError('Expected exactly train/valid/test groups')
    import csv
    assignment_path = Path(__file__).resolve().parents[1] / 'assignments.csv'
    if sha(assignment_path) != spec['meta']['assignments_sha256']:
        raise RuntimeError('Frozen assignment hash mismatch')
    with assignment_path.open() as handle:
        membership = {record['balance_row_uid'].removeprefix('catapro:'): record['new_split'] for record in csv.DictReader(handle)}
    if spec['meta']['scenario'] != 'cluster40' or len(membership) != 27658:
        raise RuntimeError('Unexpected scenario/population')
    seen = set()
    for group, rows in groups.items():
        for source, sid in rows:
            key = (source, str(sid))
            if key in seen:
                raise RuntimeError('Repeated row across/in rowspec groups')
            seen.add(key)
            row = store.get(*key)
            if membership[str(sid)] != group:
                raise RuntimeError('Frozen cluster40 membership changed')
            resolver.path(row['sequence'])
    if seen != {('catapro', sid) for sid in store.indices}:
        raise RuntimeError('Rowspec does not cover exact feature-store population')
