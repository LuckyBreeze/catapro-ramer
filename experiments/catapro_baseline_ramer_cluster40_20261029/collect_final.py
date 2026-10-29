"""Read-only collection and independent stdlib audit; no model execution."""
import base64, csv, hashlib, io, json, math, statistics, struct
from pathlib import Path
import manage

ROOT=Path(__file__).resolve().parent
def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def fp32(x): return struct.unpack('f',struct.pack('f',float(x)))[0]

bundle=manage.remote(r'''
import base64,csv,hashlib,json
from datetime import datetime,timezone
base=root.parent/'catapro_cluster40_six_seed668_20261025'
names=['prepare_result.json','review_cpu.json','review_cuda.json','preflight/preflight.json',
 'run_seed668/protocol.json','run_seed668/history.json','run_seed668/summary.json','run_seed668/status.json',
 'run_seed668/valid_predictions.csv','run_seed668/test_predictions.csv','result_audit.json',
 'execution_status.json','inputs/feature_store.json','logs/train.3184180.out','logs/train.3184180.err']
files={}
def add(name,path):
    data=path.read_bytes()
    files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())
for name in names:add(name,root/name)
add('baseline_test_predictions.csv',base/'catapro_baseline/run_seed668/test_predictions.csv')
manifest=json.loads((root/'deployment_manifest.json').read_text())
actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in manifest}
assert actual==manifest
raw=(base/'inputs/rows.csv').read_bytes()
rows=list(csv.DictReader(raw.decode().splitlines()))
labels={r['author_row_id']:r['log10_kcat'] for r in rows}
checkpoint=hashlib.sha256((root/'run_seed668/checkpoints/bestmodel.pth').read_bytes()).hexdigest()
print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),files=files,source_sha256=actual,
 rows_sha256=hashlib.sha256(raw).hexdigest(),labels=labels,checkpoint_sha256=checkpoint)))
''')
dest=ROOT/'received_final'
dest.mkdir(exist_ok=True)
for name,rec in bundle['files'].items():
    data=base64.b64decode(rec['data']); assert sha(data)==rec['sha256']
    path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert path.read_bytes()==data,name
    else:path.write_bytes(data)
receipt={k:v for k,v in bundle.items() if k not in ['files','labels']}
receipt['download_sha256']={k:v['sha256'] for k,v in bundle['files'].items()}
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
assert bundle['source_sha256']==read(ROOT/'deployment_manifest.json')
for name,digest in bundle['source_sha256'].items(): assert sha((ROOT/name).read_bytes())==digest,name
assert bundle['rows_sha256']=='a0f15c13c715006513040387a5de600fdffc42481af5830277e683d5a6dc1710'
hist=read(dest/'run_seed668/history.json');summary=read(dest/'run_seed668/summary.json')
protocol=read(dest/'run_seed668/protocol.json');ref=read(ROOT/'reference_protocol.json')
assert protocol['parameters']==ref['parameters'] and protocol['source_sha256']==ref['source_sha256']
for value in protocol['inputs'].values():
    assert value['feature_sha256']==read(dest/'prepare_result.json')['feature_sha256']
    assert value['rowspec_sha256']==sha((ROOT/'inputs/rowspec.json').read_bytes())
anchor=None;counter=0;best=None
for i,r in enumerate(hist):
    loss=r['valid']['batch_average_RMSELoss'];improved=i==0 or loss<anchor-.001
    if improved:anchor=loss;best=i;counter=1 if i==0 else 0
    else:counter+=1
    assert r['epoch']==i and r['is_best']==improved and r['early_stopping']==(counter==20)
    assert counter!=20 or i==len(hist)-1
assert counter==20 and best==summary['best_epoch_zero_based'] and len(hist)==summary['epochs_ran']
spec=read(ROOT/'inputs/rowspec.json')
def predictions(path):return list(csv.DictReader(path.open(encoding='utf-8',newline='')))
def metrics(rows):
    y=[fp32(r['log_Kcat']) for r in rows];p=[float(r['prediction']) for r in rows]
    assert all(math.isfinite(v) for v in y+p)
    err=[(a-b)**2 for a,b in zip(y,p)];mse=statistics.mean(err)
    ym=statistics.mean(y);pm=statistics.mean(p)
    corr=sum((a-ym)*(b-pm) for a,b in zip(y,p))/math.sqrt(sum((a-ym)**2 for a in y)*sum((b-pm)**2 for b in p))
    batch=statistics.mean(math.sqrt(statistics.mean(err[j:j+8])+1e-6) for j in range(0,len(err),8))
    return dict(RMSE=math.sqrt(mse),MAE=statistics.mean(abs(a-b) for a,b in zip(y,p)),R2=1-mse/statistics.pvariance(y),PCC=corr,batch_RMSELoss=batch)
calculated={}
for split in ['valid','test']:
    rows=predictions(dest/('run_seed668/'+split+'_predictions.csv'))
    expected=[uid for _,uid in (spec['eval']['test'] if split=='test' else spec['valid'])]
    assert [r['row_uid'] for r in rows]==expected and len(set(expected))==len(expected)
    for r in rows: assert fp32(r['log_Kcat'])==fp32(bundle['labels'][r['row_uid']])
    m=metrics(rows);calculated[split]=m
    for k in ['RMSE','MAE','R2','PCC']:assert abs(m[k]-summary['results'][split]['metrics'][k])<3e-6
    if split=='valid':
        assert abs(m['batch_RMSELoss']-hist[best]['valid']['batch_average_RMSELoss'])<3e-6
        for k in ['RMSE','R2','PCC']: assert abs(m[k]-hist[best]['valid'][k])<3e-6
baseline=predictions(dest/'baseline_test_predictions.csv')
test=predictions(dest/'run_seed668/test_predictions.csv')
assert [(r['row_uid'],fp32(r['log_Kcat'])) for r in baseline]==[(r['row_uid'],fp32(r['log_Kcat'])) for r in test]
base_metrics=metrics(baseline)
assert abs(base_metrics['RMSE']-1.256490612280)<1e-10
audit=read(dest/'result_audit.json')
assert audit['status']=='passed' and audit['checkpoint_sha256']==bundle['checkpoint_sha256']
for split in ['valid','test']: assert audit['prediction_sha256'][split]==receipt['download_sha256']['run_seed668/'+split+'_predictions.csv']
err=(dest/'logs/train.3184180.err').read_text()
assert not any(word in err for word in ['Traceback','RuntimeError','AssertionError','CUDA error'])
result=dict(status='passed',downloaded_files=len(bundle['files']),frozen_sources_verified=len(bundle['source_sha256']),
 best_epoch=best+1,epochs=len(hist),metrics=calculated,baseline_test=base_metrics,
 delta_RMSE=calculated['test']['RMSE']-base_metrics['RMSE'],
 relative_RMSE_increase_percent=100*(calculated['test']['RMSE']/base_metrics['RMSE']-1),
 checkpoint_sha256=bundle['checkpoint_sha256'],test_rows=len(test),
 scope='Independent saved-prediction arithmetic; no checkpoint load or new inference; selected valid aggregate metrics agree with history')
(ROOT/'completed_local_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
