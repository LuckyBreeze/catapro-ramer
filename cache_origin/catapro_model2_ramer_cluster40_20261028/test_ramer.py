import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import extract_ramer
import ramer_cache
from extract_tokens import canonical_sha, file_sha


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, padding=False, truncation=False)
        ids = [2 if x == 'J' else 3 for x in text.split()]
        return dict(input_ids=ids, attention_mask=[1] * len(ids))


class RamerChecks(unittest.TestCase):
    def test_tokenizer_keeps_unknown_residue_and_no_eos(self):
        item = dict(key='example', input='ACJX', original_lengths=[4])
        r = extract_ramer.tokenize(Tokenizer(), item)
        self.assertEqual(r['metadata']['shape'], [4, 1024])
        self.assertEqual(r['metadata']['unknown_token_count'], 1)
        self.assertEqual(r['metadata']['content_token_length'], 4)

    def test_rejects_token_count_drift(self):
        class Bad(Tokenizer):
            def __call__(self, text, **kwargs):
                return dict(input_ids=[3, 3], attention_mask=[1, 1])
        with self.assertRaises(RuntimeError):
            extract_ramer.tokenize(Bad(), dict(key='example', input='ACD', original_lengths=[3]))

    def test_rejects_eos_or_masked_residue(self):
        for ids, mask in [([3, 1], [1, 1]), ([3, 3], [1, 0])]:
            class Bad(Tokenizer):
                def __call__(self, text, **kwargs):
                    return dict(input_ids=ids, attention_mask=mask)
            with self.assertRaises(RuntimeError):
                extract_ramer.tokenize(Bad(), dict(key='x', input='AC', original_lengths=[2]))

    def fixture(self, tmp):
        root = Path(tmp)
        key = hashlib.sha256(b'ACD').hexdigest()
        item = dict(key=key, input='ACD', original_lengths=[3])
        inp = root / 'items.json'
        inp.write_text(json.dumps(dict(items=[item])))
        cfg = dict(protein_items_count=1)
        (root / 'trial_config.json').write_text(json.dumps(cfg))
        array = np.zeros((3, 1024), dtype=np.float32)
        np.save(root / (key + '.npy'), array)
        record = extract_ramer.tokenize(Tokenizer(), item)['metadata']
        metadata = {key: copy.deepcopy(record)}
        record['sha256'] = file_sha(root / (key + '.npy'))
        provenance = dict(encoder='RAMER', source_sha256={'x': 'test'},
                          source_items_sha256=file_sha(inp), config=cfg,
                          token_metadata_sha256=canonical_sha(metadata))
        meta = dict(schema_version=1, status='complete', kind='protein', provenance=provenance,
                    model_loading_info=dict(strict_complete=True), longest_preflight=dict(passed=True),
                    records={key: record})
        return root, inp, key, meta

    def verify_fixture(self, root, inp, meta):
        (root / 'manifest.json').write_text(json.dumps(meta))
        with patch.object(ramer_cache, 'ROOT', root), patch.object(ramer_cache, 'source_sha', return_value={'x': 'test'}):
            return ramer_cache.verify_ramer_cache(root, inp)

    def test_cache_accepts_complete_finite_residue_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, inp, key, meta = self.fixture(tmp)
            self.assertEqual(set(self.verify_fixture(root, inp, meta)), {key})

    def test_cache_rejects_provenance_or_loading_failure(self):
        for corruption in ['source', 'strict', 'preflight', 'coverage']:
            with tempfile.TemporaryDirectory() as tmp:
                root, inp, key, meta = self.fixture(tmp)
                if corruption == 'source': meta['provenance']['encoder'] = 'ProtT5'
                elif corruption == 'strict': meta['model_loading_info']['strict_complete'] = False
                elif corruption == 'preflight': meta['longest_preflight']['passed'] = False
                else: meta['records'] = {}
                with self.assertRaises(RuntimeError):
                    self.verify_fixture(root, inp, meta)

    def test_cache_rejects_changed_or_nonfinite_array(self):
        for update_hash in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                root, inp, key, meta = self.fixture(tmp)
                np.save(root / (key + '.npy'), np.full((3, 1024), np.nan, dtype=np.float32))
                if update_hash: meta['records'][key]['sha256'] = file_sha(root / (key + '.npy'))
                with self.assertRaises(ValueError):
                    self.verify_fixture(root, inp, meta)

    def test_cache_rejects_pooled_array_even_with_updated_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, inp, key, meta = self.fixture(tmp)
            np.save(root / (key + '.npy'), np.zeros(1024, dtype=np.float32))
            meta['records'][key]['sha256'] = file_sha(root / (key + '.npy'))
            with self.assertRaises(ValueError):
                self.verify_fixture(root, inp, meta)


if __name__ == '__main__':
    unittest.main(verbosity=2)
