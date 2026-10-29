"""Exclusive deployment, gated sbatch submission, and read-only snapshots."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE='/share/home/u17015/data/GraphKcat/'+ROOT.name
PY='/share/home/u17015/.conda/envs/graphkcat_env/bin/python'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','u17015']


def remote(code,data=None):
    code='from pathlib import Path\nroot=Path('+repr(REMOTE)+')\n'+code
    result=subprocess.run(SSH+[PY+' -B -c '+shlex.quote(code)],input=data,capture_output=True,check=True)
    return json.loads(result.stdout.decode('utf-8'))


def write(name,value,exclusive=False):
    path=ROOT/name
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def deploy():
    if (ROOT/'deployment_manifest.json').exists(): raise FileExistsError('Deployment already frozen')
    names=sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file())
    hashes={}
    for name in names:
        path=ROOT/name
        assert path.suffix in ('.py','.json','.csv','.sbatch'),name
        if path.suffix=='.py': ast.parse(path.read_text(encoding='utf-8'),filename=name)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write('deployment_manifest.json',hashes,True)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tf:
        for name in names+['deployment_manifest.json']: tf.add(ROOT/name,arcname=name,recursive=False)
    result=remote(r'''
import hashlib,io,json,sys,tarfile
if root.exists(): raise FileExistsError('New experiment root already exists')
root.mkdir()
(root/'logs').mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tf:
    for member in tf.getmembers():
        target=(root/member.name).resolve()
        if not member.isfile() or not target.is_relative_to(root.resolve()): raise ValueError(member.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(tf.extractfile(member).read())
manifest=json.loads((root/'deployment_manifest.json').read_text())
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
base=root.parent/'catapro_cluster40_six_seed668_20261025'
ident=json.loads((root/'source_identity.json').read_text())
mapping={'catapro_baseline/runner.py':'catapro_baseline/runner.py','catapro_baseline/tail_batch.py':'catapro_baseline/tail_batch.py',
 'runtime/master_store.py':'runtime/master_store.py','runtime/support.py':'runtime/support.py','runtime/residue_cache.py':'runtime/residue_cache.py',
 'author/training/kcat/model.py':'catapro_baseline/author/training/kcat/model.py','author/training/kcat/utils.py':'catapro_baseline/author/training/kcat/utils.py',
 'assignments.csv':'assignments.csv','inputs/rowspec.json':'inputs/rowspec.json','reference_protocol.json':'catapro_baseline/run_seed668/protocol.json'}
for name,original in mapping.items():
    assert hashlib.sha256((base/original).read_bytes()).hexdigest()==ident[name]['original_sha256'],original
print(json.dumps(dict(status='passed',root=str(root),files=len(manifest),original_sources_verified=len(mapping))))
''',buffer.getvalue())
    write('deployment_receipt.json',result,True)
    print(json.dumps(result))


def submit(kind):
    script='prepare.sbatch' if kind=='cpu' else 'train.sbatch'
    local_name='submission_'+kind+'.json'
    attempt=ROOT/('submission_'+kind+'_attempt.json')
    if not attempt.exists(): write(attempt.name,dict(kind=kind,script=script,utc=datetime.now(timezone.utc).isoformat()),True)
    code='kind='+repr(kind)+'\nscript='+repr(script)+'\n'+r'''
import hashlib,json,subprocess
from datetime import datetime,timezone
receipt=root/('submission_'+kind+'.json')
if receipt.exists():
    print(receipt.read_text())
else:
    for name,value in json.loads((root/'deployment_manifest.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
    if kind=='gpu':
        for name in ['prepare_result.json','review_cpu.json']:
            assert json.loads((root/name).read_text())['status']=='passed',name
        cpu=json.loads((root/'submission_cpu.json').read_text())['job_id']
        accounting=subprocess.run(['sacct','-j',cpu,'--format=JobIDRaw,State,ExitCode','--parsable2','--noheader'],text=True,capture_output=True,check=True).stdout
        assert cpu+'|COMPLETED|0:0' in accounting,accounting
    attempt=root/('submission_'+kind+'_pending.json')
    with attempt.open('x') as f:json.dump(dict(script=script,utc=datetime.now(timezone.utc).isoformat()),f)
    run=subprocess.run(['sbatch','--parsable',script],cwd=root,text=True,capture_output=True,check=True)
    job=run.stdout.strip().split(';')[0]
    assert job.isdigit(),run.stdout
    value=dict(job_id=job,kind=kind,script=script,utc=datetime.now(timezone.utc).isoformat(),stdout=run.stdout,stderr=run.stderr)
    receipt.write_text(json.dumps(value,indent=2))
    value['squeue']=subprocess.run(['squeue','-j',job,'-o','%.18i %.10P %.10T %.12M %R'],capture_output=True,text=True,check=True).stdout
    print(json.dumps(value))
'''
    result=remote(code)
    write(local_name,result)
    print(json.dumps(result))


def snapshot():
    result=remote(r'''
import json,subprocess
from datetime import datetime,timezone
files={}
names=['submission_cpu.json','submission_gpu.json','execution_status.json','pipeline_status.json','prepare_progress.json','prepare_result.json','review_cpu.json','review_cuda.json','preflight/preflight.json','run_seed668/protocol.json','run_seed668/status.json','run_seed668/history.json','run_seed668/summary.json','result_audit.json']
for name in names:
    if (root/name).is_file():files[name]=json.loads((root/name).read_text())
jobs=[files[name]['job_id'] for name in ['submission_cpu.json','submission_gpu.json'] if name in files]
logs={}
for path in (root/'logs').glob('*'):
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-5000));logs[path.name]=f.read().decode('utf-8','replace')
value=dict(utc=datetime.now(timezone.utc).isoformat(),files=files,logs=logs)
if jobs:
    value['squeue']=subprocess.run(['squeue','-j',','.join(jobs),'-h','-o','%i|%T|%M|%R'],capture_output=True,text=True).stdout
    value['sacct']=subprocess.run(['sacct','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End','--parsable2'],capture_output=True,text=True,check=True).stdout
print(json.dumps(value))
''')
    write('latest_snapshot.json',result)
    compact=dict(utc=result['utc'],squeue=result.get('squeue'),sacct=result.get('sacct'))
    for name,value in result['files'].items():
        if name not in ['run_seed668/history.json','run_seed668/protocol.json','run_seed668/summary.json']:
            compact[name]=value
    history=result['files'].get('run_seed668/history.json',[])
    if history:compact['last_epoch']=history[-1]
    compact['logs']=result['logs']
    print(json.dumps(compact,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['deploy','submit_cpu','submit_gpu','snapshot'])
    name=parser.parse_args().command
    {'deploy':deploy,'submit_cpu':lambda:submit('cpu'),'submit_gpu':lambda:submit('gpu'),'snapshot':snapshot}[name]()
