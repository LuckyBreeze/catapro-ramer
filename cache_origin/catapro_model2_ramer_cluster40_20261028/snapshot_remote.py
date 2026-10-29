"""One read-only snapshot; no scheduler mutation or new inference."""
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

root = Path(__file__).resolve().parent
report = dict(utc=datetime.now(timezone.utc).isoformat(), files={}, logs={})
for name in ['submission.json', 'extract_status.json', 'run_seed668/status.json',
             'run_seed668/protocol.json', 'run_seed668/history.json',
             'run_seed668/preflight/preflight.json', 'run_seed668/selection_lock.json',
             'run_seed668/test/status.json', 'run_seed668/test/summary.json',
             'run_seed668/test/valid_replay.json']:
    path = root / name
    if path.exists():
        report['files'][name] = json.loads(path.read_text())
path = root / 'cache' / 'ramer_protein' / 'manifest.json'
if path.exists():
    value = json.loads(path.read_text())
    report['cache'] = {k: v for k, v in value.items() if k != 'records'}
    report['cache']['completed_records'] = len(value['records'])
for path in (root / 'logs').glob('*'):
    if path.suffix in ('.out', '.err'):
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 12000))
            report['logs'][path.name] = f.read().decode('utf-8', errors='replace')
report['squeue'] = subprocess.run(['squeue', '-u', 'u17015', '-h', '-o', '%i|%j|%T|%M|%N|%R'],
                                  capture_output=True, text=True).stdout
jobs = report['files'].get('submission.json', {}).get('jobs', {})
if jobs:
    report['sacct'] = subprocess.run(['sacct', '-j', ','.join(jobs.values()), '--noheader', '-P',
        '--format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Start,End'], capture_output=True, text=True).stdout
print(json.dumps(report))
