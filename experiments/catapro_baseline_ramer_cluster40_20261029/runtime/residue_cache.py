"""Residue resolver over the original audited cache + a supplementary cache.

The original cache (catapro_cond_esm2_cnn_20261010/cache) covers the 6,437
unique sequences of the fixed kcat split. The self-made splits contain 2,256
additional unique sequences; this module extracts them with the SAME frozen
ESM-2 3B layer-36 protocol (fp16, BOS/EOS removed, UZOB->X, no truncation)
into a separate supplementary cache, never touching the original.

Extract:
  python residue_cache.py --extract --sequences-json missing_sequences.json \
    --model-pt <esm2_t36_3B_UR50D.pt> --cache-ext <dir>
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from support import sha, write_json

MODEL_SHA = '7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500'


def normalize_sequence(raw):
    if not isinstance(raw, str) or not raw or len(raw) > 1022:
        raise RuntimeError('Missing/overlength sequence: no truncation is authorized')
    seq = re.sub('[UZOB]', 'X', raw)
    if not re.fullmatch('[A-Z]+', seq):
        raise RuntimeError('Invalid sequence characters')
    return seq


class ResidueResolver:
    """sequence -> per-residue npy path (original cache first, then ext)."""

    def __init__(self, cache_dir, cache_ext=None):
        meta = json.loads((Path(cache_dir) / 'order.json').read_text())
        self.table = {seq: str(Path(cache_dir) / 'residues' / f'{i:06d}.npy')
                      for i, seq in enumerate(meta['sequences'])}
        if cache_ext is not None:
            order_file = Path(cache_ext) / 'ext_order.json'
            ext = json.loads(order_file.read_text()) if order_file.exists() \
                else {'sequences': []}
            for i, seq in enumerate(ext['sequences']):
                if seq in self.table:
                    raise RuntimeError('Ext cache duplicates original sequence')
                self.table[seq] = str(Path(cache_ext) / 'residues' / f'ext_{i:06d}.npy')

    def missing(self, sequences):
        return sorted({normalize_sequence(s) for s in sequences} - set(self.table))

    def path(self, sequence):
        return self.table[sequence]

    def __len__(self):
        return len(self.table)


def extract(sequences, model_pt, cache_ext):
    import torch
    if sha(model_pt) != MODEL_SHA:
        raise RuntimeError('ESM checkpoint hash mismatch')
    cache = Path(cache_ext)
    (cache / 'residues').mkdir(parents=True, exist_ok=True)
    seqs = sorted({normalize_sequence(s) for s in sequences})
    order_path = cache / 'ext_order.json'
    if order_path.exists():
        existing = json.loads(order_path.read_text())
        if existing['sequences'] != seqs:
            raise RuntimeError('Ext cache order differs; refusing overwrite')
    else:
        write_json(order_path, dict(sequences=seqs,
                                    preprocessing='UZOB->X; full length; no truncation; no BOS/EOS',
                                    model_sha256=MODEL_SHA, layer=36, dimension=2560,
                                    dtype='float16'))
    import esm
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_pt)
    model = model.half().cuda().eval()
    converter = alphabet.get_batch_converter()
    started = time.monotonic()
    longest = max(range(len(seqs)), key=lambda i: len(seqs[i]))
    order = [longest] + [i for i in range(len(seqs)) if i != longest]
    for done, uid in enumerate(order, 1):
        seq = seqs[uid]
        path = cache / 'residues' / f'ext_{uid:06d}.npy'
        if path.exists():
            values = np.load(path, allow_pickle=False)
        else:
            _, strings, tokens = converter([(str(uid), seq)])
            if strings != [seq]:
                raise RuntimeError('Tokenizer changed input')
            with torch.inference_mode():
                out = model(tokens.cuda(), repr_layers=[36], return_contacts=False)
                values = out['representations'][36][0, 1:len(seq) + 1].cpu().numpy().copy()
            del out, tokens
            if values.shape != (len(seq), 2560) or not np.isfinite(values).all():
                raise RuntimeError('Non-finite/misaligned ESM features')
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.save(f, values, allow_pickle=False)
            os.replace(tmp, path)
        if values.dtype != np.float16 or values.shape != (len(seq), 2560):
            raise RuntimeError('Invalid existing ext residue file')
        if done == 1 or done % 25 == 0 or done == len(seqs):
            progress = dict(status='extracting', completed=done, total=len(seqs),
                            elapsed_seconds=time.monotonic() - started,
                            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
            write_json(cache / 'ext_progress.json', progress)
            print(json.dumps(progress), flush=True)
    write_json(cache / 'ext_manifest.json',
               dict(status='complete', layer=36, dimension=2560, model_sha256=MODEL_SHA,
                    unique_sequences=len(seqs), dtype='float16',
                    elapsed_seconds=time.monotonic() - started))
    write_json(cache / 'ext_progress.json', dict(status='complete', completed=len(seqs),
                                                 total=len(seqs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extract', action='store_true')
    p.add_argument('--sequences-json', required=True)
    p.add_argument('--model-pt', required=True)
    p.add_argument('--cache-ext', required=True)
    a = p.parse_args()
    if a.extract:
        extract(json.loads(Path(a.sequences_json).read_text()), a.model_pt, a.cache_ext)


if __name__ == '__main__':
    main()
