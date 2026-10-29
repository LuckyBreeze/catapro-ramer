"""RAMER protein-cache provenance and per-array validation."""
import json
from pathlib import Path

from extract_tokens import canonical_sha, file_sha, verify_array

ROOT = Path(__file__).resolve().parent


def source_sha():
    paths = ['extract_ramer.py', 'ramer_cache.py', 'extract_tokens.py',
             'trial_config.json', 'resource_identity.json', 'upstream_sha256.json']
    official = json.loads((ROOT / 'upstream_sha256.json').read_text(encoding='utf-8'))
    for name, expected in official.items():
        if file_sha(ROOT / name) != expected:
            raise RuntimeError('Official RAMER source changed: ' + name)
    return {name: file_sha(ROOT / name) for name in paths + sorted(official)}


def verify_ramer_cache(root, items_path):
    import numpy as np
    root, items_path = Path(root), Path(items_path)
    meta = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    cfg = json.loads((ROOT / 'trial_config.json').read_text(encoding='utf-8'))
    doc = json.loads(items_path.read_text(encoding='utf-8'))
    items = {r['key']: r for r in doc['items']}
    if len(items) != cfg['protein_items_count'] or len(items) != len(doc['items']):
        raise RuntimeError('Wrong RAMER input population')
    if (meta.get('schema_version') != 1 or meta.get('status') != 'complete'
            or meta.get('kind') != 'protein'):
        raise RuntimeError('RAMER protein cache is incomplete or wrong kind')
    provenance = meta['provenance']
    if (provenance.get('encoder') != 'RAMER' or provenance.get('source_sha256') != source_sha()
            or provenance.get('source_items_sha256') != file_sha(items_path)
            or provenance.get('config') != cfg):
        raise RuntimeError('RAMER cache source/config/input provenance mismatch')
    if not meta.get('model_loading_info', {}).get('strict_complete'):
        raise RuntimeError('RAMER weights were not completely loaded')
    if not meta.get('longest_preflight', {}).get('passed'):
        raise RuntimeError('RAMER longest GPU probe did not pass')
    records = meta['records']
    if set(records) != set(items):
        raise RuntimeError('RAMER cache coverage mismatch')
    for k, rec in records.items():
        item = items[k]
        length = len(item['input'])
        expected = dict(key=k, file=k + '.npy', shape=[length, 1024], dtype='float32',
                        input_length=length, original_lengths=item['original_lengths'],
                        content_token_length=length,
                        token_ids_sha256=rec['token_ids_sha256'],
                        unknown_token_count=rec['unknown_token_count'])
        if k != __import__('hashlib').sha256(item['input'].encode()).hexdigest():
            raise RuntimeError('RAMER input key mismatch')
        verify_array(root / rec['file'], rec, expected)
    if canonical_sha({k: {n: v for n, v in r.items() if n != 'sha256'}
                      for k, r in records.items()}) != provenance['token_metadata_sha256']:
        raise RuntimeError('RAMER token metadata digest mismatch')
    return records
