"""Submit the authorized extraction -> original-model2 training/test chain once."""
import fcntl
import json
from pathlib import Path
import subprocess
import uuid

from extract_tokens import atomic_json, utc_now
from verify_deployment import verify

ROOT = Path(__file__).resolve().parent


def main():
    verify()
    with (ROOT / '.submission.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = ROOT / 'submission.json'
        if path.exists():
            raise RuntimeError('Submission record exists; inspect it instead of resubmitting')
        (ROOT / 'logs').mkdir(exist_ok=True)
        record = dict(utc=utc_now(), jobs={}, pending=None, scenario='cluster40', seed=668,
                      selection='original batch-average RMSELoss', encoder='RAMER')
        atomic_json(path, record)
        for stage in ['protein', 'train']:
            tag = 'm2_ramer_' + stage + '_' + uuid.uuid4().hex
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', '--comment=' + tag]
            if stage == 'train':
                command.append('--dependency=afterok:' + record['jobs']['protein'])
            command.append(stage + '.sbatch')
            record['pending'] = dict(stage=stage, unique_comment=tag, command=command)
            atomic_json(path, record)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Ambiguous Slurm response: ' + result.stdout)
            record['jobs'][stage] = job
            record['pending'] = None
            atomic_json(path, record)
            print(json.dumps(dict(stage=stage, job=job)), flush=True)
        subprocess.run(['squeue', '-j', ','.join(record['jobs'].values()),
                        '-o', '%.18i %.22j %.12T %.10M %.35R'], check=True)


if __name__ == '__main__':
    main()
