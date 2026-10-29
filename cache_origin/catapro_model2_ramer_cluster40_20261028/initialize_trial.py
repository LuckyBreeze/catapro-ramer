"""Prepare a new trial locally; never modify the earlier model2 experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'catapro_residue_query_model2_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    names = ['model.py', 'author_utils.py', 'tail_batch.py', 'prepare.py',
             'extract_tokens.py', 'checkpoints.json', 'test_model.py',
             'test_extract_tokens.py', 'test_train.py']
    unchanged = {}
    for name in names:
        with (ROOT / name).open('xb') as f:
            f.write((OLD / name).read_bytes())
        unchanged[name] = sha(ROOT / name)
    # The only data-reader adaptation is the RAMER protein-cache identity contract.
    data = (OLD / 'data.py').read_text(encoding='utf-8')
    original = 'def verify_cache(root, items_path, kind):\n'
    assert data.count(original) == 1
    data = data.replace(original, original +
        "    if kind == 'protein':\n" +
        "        from ramer_cache import verify_ramer_cache\n" +
        "        return verify_ramer_cache(root, items_path)\n")
    with (ROOT / 'data.py').open('x', encoding='utf-8', newline='\n') as f:
        f.write(data)
    # Add provenance files; all training, selection, evaluation and model functions stay intact.
    train = (OLD / 'train.py').read_text(encoding='utf-8')
    original = '"prepare.py", "extract_tokens.py")'
    assert train.count(original) == 1
    train = train.replace(original, '"prepare.py", "extract_tokens.py", "extract_ramer.py",\n'
                          '             "ramer_cache.py", "trial_config.json", "resource_identity.json",\n'
                          '             "upstream/Data2seq/proteinT5_embedding.py",\n'
                          '             "upstream/Data2seq/Data2seq.py")')
    with (ROOT / 'train.py').open('x', encoding='utf-8', newline='\n') as f:
        f.write(train)
    old_identity = json.loads((ROOT.parent / 'catapro_model2_r2_selection_cluster40_20261028' /
                               'baseline_identity.json').read_text(encoding='utf-8'))
    save('baseline_identity.json', {
        'base_local': str(OLD),
        'base_remote': '/share/home/u17015/data/GraphKcat/' + OLD.name,
        'new_remote': '/share/home/u17015/data/GraphKcat/' + ROOT.name,
        'copied_unchanged': unchanged,
        'old_data_sha256': sha(OLD / 'data.py'),
        'old_train_sha256': sha(OLD / 'train.py'),
        'old_inputs': old_identity['baseline_inputs'],
        'old_formal_initial_sha256': old_identity['baseline_formal_initial_sha256'],
        'old_deployment_manifest_sha256': old_identity['baseline_deployment_manifest_sha256']})
    save('trial_config.json', {
        'authorized': True, 'scenario': 'cluster40', 'seed': 668,
        'protein_encoder': 'RAMER official protein-sequence inference branch',
        'representation': 'LoRA-adapted last_hidden_state before pooling; no learned linear projection',
        'upstream_commit': '5f453a0f836b407c77ff666acb2f9dde7c595a30',
        'hf_revision': '4bc32f7b9fb5d686717c3e7e1462d691efcc59c5',
        'weights_root': '/share/home/u17015/data/RAMER',
        'extract_env': '/share/home/u17015/.conda/envs/cpdiffusion',
        'train_env': '/share/home/u17015/.conda/envs/graphkcat_env',
        'protein_items_count': 13192, 'width': 1024, 'dtype': 'float32',
        'preprocessing': 'reuse original prepared protein_items; >1000 first500+last500; UZOB->X',
        'tokenization': 'official legacy=False, add_special_tokens=False, no truncation',
        'extract_batch_size': 1, 'freeze_encoder': True,
        'selection': 'author_batch_average_RMSELoss_raw',
        'selection_confirmation': '沿用最初模型2：逐批 RMSELoss 的均值最低',
        'early_stopping': 'original EarlyStopping; min_delta0.001; patience20; first counter1; max150',
        'loss_and_model': 'unchanged original model2 RMSELoss and architecture',
        'test_policy': 'original locked model, full valid replay then one test',
        'other_experiments': 'no writes or cancellations',
        'dependency_policy': 'compatibility checks first; no dependency installation or environment mutation'})
    upstream = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                for p in sorted((ROOT / 'upstream').rglob('*')) if p.is_file()}
    save('upstream_sha256.json', upstream)
    hf = json.loads((ROOT / 'hf_model_index.json').read_text(encoding='utf-8-sig'))
    needed = {'model/RAMER.pth', 'model/ProtT5/pytorch_model.bin',
              'model/ProtT5/config.json', 'model/ProtT5/tokenizer_config.json',
              'model/ProtT5/special_tokens_map.json', 'model/ProtT5/spiece.model'}
    files = {r['path']: dict(bytes=r['size'], sha256=r.get('lfs', {}).get('oid'),
                            git_blob_sha1=r['oid'] if 'lfs' not in r else None)
             for r in hf if r['path'] in needed}
    assert set(files) == needed
    save('resource_identity.json', {'hf_revision': '4bc32f7b9fb5d686717c3e7e1462d691efcc59c5',
                                    'files': files})
    print(json.dumps({'status': 'prepared', 'unchanged_files': len(unchanged),
                      'official_source_files': len(upstream)}))


if __name__ == '__main__':
    main()
