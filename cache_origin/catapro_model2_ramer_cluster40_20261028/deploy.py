"""Exclusive deployment to a new directory; no original files are written."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = '/share/home/u17015/data/GraphKcat/' + ROOT.name
PY = '/share/home/u17015/.conda/envs/graphkcat_env/bin/python'
SSH = ['C:/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=30']
SCP = ['C:/Windows/System32/OpenSSH/scp.exe', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def remote(source):
    return subprocess.run(SSH + ['u17015', shlex.quote(PY) + ' -B -'], input=source,
                          capture_output=True, text=True, encoding='utf-8')


def build():
    checks = json.loads((ROOT / 'local_checks.json').read_text())
    assert checks['status'] == 'passed'
    for name, digest in checks['reviewed_sha256'].items():
        assert sha(ROOT / name) == digest, name
    required = ['trial_config.json', 'baseline_identity.json', 'resource_identity.json',
                'upstream_sha256.json', 'environment_audit.json', 'local_checks.json',
                'checkpoints.json', 'hf_model_index.json', 'upstream_commit.json',
                'checkpoint_metadata.json']
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sbatch'))
    paths += [ROOT / name for name in required]
    paths += [p for p in (ROOT / 'upstream').rglob('*') if p.is_file()]
    manifest = {}
    for p in paths:
        assert not p.is_symlink()
        name = p.relative_to(ROOT).as_posix()
        assert name not in manifest
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=name)
        manifest[name] = sha(p)
    save('deployment_manifest.json', manifest)
    archive = ROOT / 'deployment.tar.gz'
    if archive.exists():
        raise FileExistsError('Deployment archive already exists')
    with tarfile.open(archive, 'w:gz') as tf:
        for name in sorted(manifest) + ['deployment_manifest.json']:
            tf.add(ROOT / name, arcname=name, recursive=False)
    return archive, manifest


def deploy():
    if (ROOT / 'deployment_attempt.json').exists():
        raise RuntimeError('Deployment was already attempted; inspect before retrying')
    archive, manifest = build()
    save('deployment_attempt.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                                       remote=REMOTE, archive_sha256=sha(archive)))
    result = remote("from pathlib import Path\nr=Path(%r)\nr.mkdir(parents=False, exist_ok=False)\n" % REMOTE)
    save('deployment_directory_response.json', dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
    subprocess.run(SCP + [str(archive), 'u17015:' + REMOTE + '/deployment.tar.gz'], check=True)
    source = '''import hashlib, tarfile, subprocess
from pathlib import Path
r=Path(REMOTE)
p=r/'deployment.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()==ARCHIVE_SHA
with tarfile.open(p,'r:gz') as tf:
    members=tf.getmembers()
    assert len(members)==len(NAMES) and {m.name for m in members}==set(NAMES)
    for m in members:
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        dst=r/m.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as out: out.write(tf.extractfile(m).read())
subprocess.run([PY,'-B','verify_deployment.py'],cwd=r,check=True)
subprocess.run(['bash','-n','protein.sbatch'],cwd=r,check=True)
subprocess.run(['bash','-n','train.sbatch'],cwd=r,check=True)
'''.replace('REMOTE', repr(REMOTE)).replace('ARCHIVE_SHA', repr(sha(archive))).replace('NAMES', repr(list(manifest) + ['deployment_manifest.json'])).replace('PY', repr(PY))
    result = remote(source)
    save('deployment_verification_response.json', dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
    receipt = dict(status='verified', remote=REMOTE, files=len(manifest),
                   manifest_sha256=sha(ROOT / 'deployment_manifest.json'),
                   archive_sha256=sha(archive), stdout=result.stdout)
    save('deployment_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def submit():
    receipt = json.loads((ROOT / 'deployment_receipt.json').read_text())
    assert receipt['status'] == 'verified' and receipt['manifest_sha256'] == sha(ROOT / 'deployment_manifest.json')
    save('submission_dispatch_attempt.json', dict(utc=datetime.now(timezone.utc).isoformat(), remote=REMOTE))
    cmd = 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.quote(PY) + ' -B submit_remote.py'
    result = subprocess.run(SSH + ['u17015', cmd], capture_output=True, text=True, encoding='utf-8')
    save('submission_dispatch_response.json', dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    result.check_returncode()
    subprocess.run(SCP + ['u17015:' + REMOTE + '/submission.json', str(ROOT / 'submission.json')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['deploy', 'submit'])
    args = parser.parse_args()
    {'deploy': deploy, 'submit': submit}[args.mode]()
