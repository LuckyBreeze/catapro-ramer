"""Run guarded preparation or the unchanged CataPro preflight/train/evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent


def status(stage,state,**extra):
    path=ROOT/'execution_status.json'
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,state=state,utc=datetime.now(timezone.utc).isoformat(),**extra),indent=2),encoding='utf-8')
    tmp.replace(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['cpu','gpu'])
    mode=parser.parse_args().mode
    if mode=='cpu':
        commands=[('prepare',['workflow.py','prepare']),('review_cpu',['workflow.py','review_cpu'])]
    else:
        for name in ['prepare_result.json','review_cpu.json']:
            assert json.loads((ROOT/name).read_text())['status']=='passed',name
        args=['catapro_baseline/runner.py','--catapro-root','author','--split-dir','inputs',
              '--feature-dir','inputs','--seed','668']
        commands=[('review_cuda',['workflow.py','review_cuda']),
                  ('original_preflight',args+['--output-dir','preflight','--preflight']),
                  ('original_train_and_test',args+['--output-dir','run_seed668']),
                  ('audit_result',['workflow.py','audit_result'])]
    stage='initializing'
    try:
        for stage,args in commands:
            status(stage,'running')
            print(json.dumps(dict(stage=stage,command=[sys.executable,'-B','-u']+args)),flush=True)
            subprocess.run([sys.executable,'-B','-u']+args,cwd=ROOT,check=True)
        status(stage,'complete',mode=mode)
    except Exception as error:
        status(stage,'failed',mode=mode,error=repr(error),traceback=traceback.format_exc())
        raise
