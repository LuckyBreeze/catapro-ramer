"""Verify the deployed scripts and read-only original model2 evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(check_baseline=True):
    manifest = json.loads((ROOT / 'deployment_manifest.json').read_text())
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Deployment source changed: ' + name)
    checks = json.loads((ROOT / 'local_checks.json').read_text())
    if checks['status'] != 'passed':
        raise RuntimeError('Local code review failed')
    for name, expected in checks['reviewed_sha256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Source changed after local checks: ' + name)
    env = json.loads((ROOT / 'environment_audit.json').read_text())
    if env['status'] != 'cpu_interface_passed_gpu_pending':
        raise RuntimeError('CPU encoder environment probe failed')
    cfg = json.loads((ROOT / 'trial_config.json').read_text())
    if not cfg['authorized'] or cfg['selection'] != 'author_batch_average_RMSELoss_raw':
        raise RuntimeError('Unexpected execution/selection configuration')
    identity = json.loads((ROOT / 'baseline_identity.json').read_text())
    if check_baseline:
        base = Path(identity['base_remote'])
        if sha(base / 'deployment_manifest.json') != identity['old_deployment_manifest_sha256']:
            raise RuntimeError('Original deployment manifest identity changed')
        old_sources = json.loads((base / 'deployment_manifest.json').read_text())
        for name, expected in old_sources.items():
            if sha(base / name) != expected:
                raise RuntimeError('Original source changed: ' + name)
        paths = dict(prepared_manifest='prepared/manifest.json', prepared_rows='prepared/rows.json',
                     prepared_maccs='prepared/maccs.npy', protein_manifest='cache/protein/manifest.json',
                     molt5_manifest='cache/molt5/manifest.json')
        for name, relative in paths.items():
            if sha(base / relative) != identity['old_inputs'][name]:
                raise RuntimeError('Original input changed: ' + name)
    print(json.dumps(dict(status='passed', deployed_files=len(manifest),
                          old_sources_and_inputs_checked=check_baseline)), flush=True)


if __name__ == '__main__':
    verify()
