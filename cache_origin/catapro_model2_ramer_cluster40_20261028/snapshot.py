import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
remote = '/share/home/u17015/data/GraphKcat/' + root.name
command = '/share/home/u17015/.conda/envs/graphkcat_env/bin/python -B ' + remote + '/snapshot_remote.py'
result = subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe', '-o', 'BatchMode=yes', '-o',
                         'ConnectTimeout=15', 'u17015', command], capture_output=True, text=True,
                        encoding='utf-8', check=True)
report = json.loads(result.stdout)
(root / 'latest_snapshot.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(utc=report['utc'], squeue=report['squeue'], sacct=report.get('sacct'),
      extraction=report['files'].get('extract_status.json'),
      training=report['files'].get('run_seed668/status.json'),
      preflight_complete='run_seed668/preflight/preflight.json' in report['files'],
      logs=report['logs']), ensure_ascii=False), flush=True)
