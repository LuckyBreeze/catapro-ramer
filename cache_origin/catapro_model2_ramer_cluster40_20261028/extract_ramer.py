"""Frozen official RAMER inference tokens; independent output, no scheduler calls."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from extract_tokens import (atomic_json, canonical_sha, commit_array, file_sha,
                            open_manifest, output_lock, read_items, save_manifest,
                            utc_now)
from ramer_cache import source_sha

ROOT = Path(__file__).resolve().parent


def emit(event, **values):
    record = dict(event=event, utc=utc_now(), **values)
    atomic_json(ROOT / 'extract_status.json', record)
    print(json.dumps(record), flush=True)


def verify_resources(weights_root):
    specification = json.loads((ROOT / 'resource_identity.json').read_text(encoding='utf-8'))
    result = {}
    for name, expected in specification['files'].items():
        path = weights_root / name
        if path.stat().st_size != expected['bytes']:
            raise RuntimeError('RAMER resource length mismatch: ' + name)
        digest = file_sha(path)
        if expected['sha256'] is not None:
            if digest != expected['sha256']:
                raise RuntimeError('RAMER official LFS SHA256 mismatch: ' + name)
        else:
            content = path.read_bytes()
            git_digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if git_digest != expected['git_blob_sha1']:
                raise RuntimeError('RAMER official Git blob mismatch: ' + name)
        result[name] = dict(bytes=path.stat().st_size, sha256=digest)
    for name in ['added_tokens.json', 'tokenizer.json', 'model.safetensors',
                 'model.safetensors.index.json', 'pytorch_model.bin.index.json']:
        if (weights_root / 'model' / 'ProtT5' / name).exists():
            raise RuntimeError('Unapproved shadow tokenizer/weight file: ' + name)
    return result


def tokenize(tokenizer, item):
    text = ' '.join(item['input'])
    encoded = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)
    ids, mask = list(encoded['input_ids']), list(encoded['attention_mask'])
    if (len(ids) != len(item['input']) or mask != [1] * len(ids)
            or any(i in (tokenizer.pad_token_id, tokenizer.eos_token_id) for i in ids)):
        raise RuntimeError('Official RAMER tokenization lost residue identity: ' + item['key'])
    return dict(ids=ids, metadata=dict(
        key=item['key'], file=item['key'] + '.npy', shape=[len(ids), 1024], dtype='float32',
        input_length=len(item['input']), original_lengths=item['original_lengths'],
        content_token_length=len(ids), token_ids_sha256=canonical_sha(ids),
        unknown_token_count=sum(i == tokenizer.unk_token_id for i in ids)))


def extract(model, item, expected):
    import numpy as np
    import torch
    # This is the author's unmodified protein-sequence forward at stage=inference.
    with torch.no_grad():
        hidden, mask = model([item['input']])
        if (hidden.dtype != torch.float32 or list(hidden.shape) != [1, len(expected['ids']), 1024]
                or list(mask.shape) != [1, len(expected['ids'])] or not bool((mask == 1).all())):
            raise RuntimeError('RAMER output shape/dtype/mask mismatch: ' + item['key'])
        value = hidden[0].detach().cpu().numpy().copy()
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise RuntimeError('RAMER output is not finite FP32')
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('RAMER extraction must run in sbatch')
    cfg = json.loads((ROOT / 'trial_config.json').read_text(encoding='utf-8'))
    if not cfg['authorized'] or cfg['extract_batch_size'] != 1:
        raise RuntimeError('Unapproved extraction configuration')
    output = ROOT / 'cache' / 'ramer_protein'
    if (output / 'manifest.json').exists():
        raise FileExistsError('Refusing to overwrite/restart an existing RAMER extraction')
    base = Path(json.loads((ROOT / 'baseline_identity.json').read_text())['base_remote'])
    items_path = base / 'prepared' / 'protein_items.json'
    prepared = json.loads((base / 'prepared' / 'manifest.json').read_text())
    if file_sha(items_path) != prepared['files']['protein_items.json']:
        raise RuntimeError('Original protein input has changed')
    items, items_sha, _ = read_items(items_path, 'protein')
    if len(items) != cfg['protein_items_count']:
        raise RuntimeError('RAMER input population differs from original model2')
    frozen_sources = source_sha()
    weights_root = Path(cfg['weights_root'])
    emit('verifying_official_resources', items=len(items))
    fingerprint = verify_resources(weights_root)

    import numpy as np
    import peft
    import sentencepiece
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA is unavailable; no CPU fallback')
    torch.set_num_threads(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(668)
    torch.cuda.manual_seed_all(668)
    torch.set_default_dtype(torch.float32)
    runtime = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                   transformers=transformers.__version__, peft=peft.__version__,
                   sentencepiece=sentencepiece.__version__, numpy=np.__version__,
                   cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   dtype='float32', tf32=False, batch_size=1)
    emit('loading_official_ramer', runtime=runtime)
    # Preserve the author's relative model path by setting CWD; original files remain read-only.
    os.chdir(weights_root)
    sys.path.insert(0, str(ROOT / 'upstream' / 'Data2seq'))
    from Data2seq import BioData2Seq
    model = BioData2Seq(modality='protein-sequence', embed_dim=1024, protein_stage='inference')
    checkpoint = torch.load(weights_root / 'model' / 'RAMER.pth', map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not all(k.startswith('module.') for k in checkpoint):
        raise RuntimeError('Unexpected RAMER checkpoint key format')
    state = {k[len('module.'):]: v for k, v in checkpoint.items()}
    if len(state) != len(checkpoint) or len(state) != 270:
        raise RuntimeError('RAMER checkpoint key collision or unexpected key count')
    actual = model.state_dict()
    if (set(state) != set(actual)
            or any(state[k].shape != actual[k].shape for k in state)
            or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values())):
        raise RuntimeError('RAMER checkpoint keys/shapes/precision/finiteness mismatch')
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError('Incomplete RAMER weights')
    lora_b = {k: int(torch.count_nonzero(v)) for k, v in state.items() if '.lora_B.' in k}
    if len(lora_b) != 36 or not any(v > 0 for v in lora_b.values()):
        raise RuntimeError('RAMER trained LoRA B weights are missing or all zero')
    model.requires_grad_(False)
    model.eval().to(device='cuda', dtype=torch.float32)
    if (any(m.training for m in model.modules())
            or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise RuntimeError('RAMER is not fully frozen eval FP32')
    loading_info = dict(strict_complete=True, keys=len(state), missing_keys=[], unexpected_keys=[],
                        trained_lora_b_nonzero=lora_b, protein_stage=model.protein_embed.stage)
    del checkpoint, state, actual
    gc.collect()
    if verify_resources(weights_root) != fingerprint:
        raise RuntimeError('RAMER resource changed during loading')
    tokenizer = model.protein_embed.tokenizer
    tokenized = {item['key']: tokenize(tokenizer, item) for item in items}
    metadata = {k: t['metadata'] for k, t in tokenized.items()}
    by_key = {item['key']: item for item in items}
    order = sorted(by_key, key=lambda k: (-len(by_key[k]['input']), k))
    provenance = dict(kind='protein', encoder='RAMER', source_items_sha256=items_sha,
                      extract_code_sha256=file_sha(ROOT / 'extract_ramer.py'),
                      source_sha256=frozen_sources, config=cfg, runtime=runtime,
                      checkpoint_files=fingerprint, items_count=len(items),
                      token_metadata_sha256=canonical_sha(metadata))
    with output_lock(output):
        manifest = open_manifest(output, provenance, metadata)
        try:
            manifest['model_loading_info'] = loading_info
            longest = order[0]
            first = extract(model, by_key[longest], tokenized[longest])
            repeated = extract(model, by_key[longest], tokenized[longest])
            if not np.array_equal(first, repeated):
                raise RuntimeError('RAMER eval output did not repeat exactly')
            # Explicit check that official inference returns unprojected last_hidden_state.
            with torch.no_grad():
                ids = torch.tensor([tokenized[longest]['ids']], device='cuda')
                direct = model.protein_embed.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0].cpu().numpy()
            if not np.array_equal(first, direct):
                raise RuntimeError('RAMER official inference representation is inconsistent')
            commit_array(output, manifest, first, metadata[longest])
            manifest['longest_preflight'] = dict(passed=True, key=longest, shape=list(first.shape),
                exact_repeat=True, official_unprojected_output_exact=True, completed_at=utc_now())
            save_manifest(output, manifest)
            emit('longest_preflight_passed', shape=list(first.shape), completed=1, total=len(items),
                 strict_weight_loading=True, official_output_verified=True)
            for k in order[1:]:
                commit_array(output, manifest, extract(model, by_key[k], tokenized[k]), metadata[k])
                count = len(manifest['records'])
                if count % 50 == 0 or count == len(items):
                    emit('extracting', completed=count, total=len(items))
            if source_sha() != frozen_sources or file_sha(items_path) != items_sha:
                raise RuntimeError('RAMER extraction source/input changed')
            manifest['status'] = 'complete'
            manifest['completed_at'] = utc_now()
            save_manifest(output, manifest)
            emit('complete', completed=len(manifest['records']), total=len(items),
                 manifest_sha256=file_sha(output / 'manifest.json'))
        except BaseException as exc:
            manifest['status'] = 'failed'
            manifest['error'] = type(exc).__name__ + ': ' + str(exc)
            save_manifest(output, manifest)
            raise


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        emit('failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
