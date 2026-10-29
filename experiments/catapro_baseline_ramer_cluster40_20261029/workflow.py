"""Independent RAMER pooled-input preparation and checks; original trainer is untouched."""
import argparse
import hashlib
import importlib.util
import io
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'catapro_cluster40_six_seed668_20261025'
TOKENS = ROOT.parent/'catapro_residue_query_model2_20261027'
RAMER = ROOT.parent/'catapro_model2_ramer_cluster40_20261028'
sys.path.insert(0, str(ROOT/'catapro_baseline'))
sys.path.insert(0, str(ROOT/'runtime'))
from master_store import MasterStore
from tail_batch import MergeSingletonBatchSampler

EXPECTED = {
    'inputs/feature_store.json':'d489a8702bd58da9af8e117da88f9ab2e18a1806af218290774eb9ad43860d8d',
    'inputs/rowspec.json':'e92f55cd7881a2a6615ebabf189bd421ced3e448a357881024501813e97667bc',
    'inputs/rows.csv':'a0f15c13c715006513040387a5de600fdffc42481af5830277e683d5a6dc1710',
    'inputs/feature1959.npy':'18ca9ea49e78b338e6d64040874c2048b4fde07457a595fd77114fd91baaba86',
    'assignments.csv':'721cc57c15ea1e87ea289118868f9d5b9007a9aaa213e45a4d76844e13d08138',
}
CACHE_SHA = '2ccc424a22c0315e67318a1175edad53ef45ec497e3cf4bf5372224aaf01caa3'
ITEMS_SHA = '812a36e67ced89f5e3bbd34e3f55094ea23daccf87b4a24b73289eb6d08abac5'
COUNTS = dict(train=21585,valid=3037,test=3036)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def source_check():
    identities = read(ROOT/'source_identity.json')
    for name, rec in identities.items():
        assert sha(ROOT/name) == rec['deployed_sha256'], name
        if name != 'runtime/master_store.py':
            assert rec['original_sha256'] == rec['deployed_sha256'], name
    raw = (ROOT/'runtime/master_store.py').read_bytes()
    assert raw.count(b"{'name': 'RAMER', 'width': 1024}") == 1
    restored = raw.replace(b"{'name': 'RAMER', 'width': 1024}",b"{'name': 'ProtT5', 'width': 1024}")
    assert hashlib.sha256(restored).hexdigest() == identities['runtime/master_store.py']['original_sha256']
    assert sha(ROOT/'catapro_baseline/runner.py') == read(ROOT/'reference_protocol.json')['source_sha256']['runner']
    deployment = read(ROOT/'deployment_manifest.json')
    for name, value in deployment.items(): assert sha(ROOT/name) == value, name
    return dict(copied_files=len(identities),deployed_files=len(deployment),runner_unchanged=True,
                model_utils_unchanged=True,sole_reader_change='protein block identity ProtT5 to RAMER')


def protein_input(seq):
    assert isinstance(seq,str) and re.fullmatch('[A-Z]+',seq)
    return re.sub('[UZOB]','X',seq[:500]+seq[-500:] if len(seq)>1000 else seq)


def pool(array):
    assert array.ndim == 2 and array.shape[1] == 1024 and array.shape[0] > 0
    assert array.dtype == np.float32 and np.isfinite(array).all()
    # Cached arrays contain only valid residues, so this is the official masked sum/count.
    return (torch.from_numpy(array).sum(dim=0)/array.shape[0]).numpy()


def prepare():
    source_check()
    output = ROOT/'inputs/feature1959.npy'
    if output.exists() or (ROOT/'inputs/feature_store.json').exists():
        raise FileExistsError('Prepared feature output already exists; no overwrite')
    for name,value in EXPECTED.items(): assert sha(BASE/name) == value,name
    assert sha(ROOT/'inputs/rowspec.json') == EXPECTED['inputs/rowspec.json']
    assert sha(ROOT/'assignments.csv') == EXPECTED['assignments.csv']
    cache = RAMER/'cache/ramer_protein'
    assert sha(cache/'manifest.json') == CACHE_SHA
    items_path = TOKENS/'prepared/protein_items.json'
    assert sha(items_path) == ITEMS_SHA
    items = {r['key']:r for r in read(items_path)['items']}
    manifest = read(cache/'manifest.json')
    assert manifest['status'] == 'complete' and manifest['provenance']['encoder'] == 'RAMER'
    assert manifest['provenance']['source_items_sha256'] == ITEMS_SHA
    assert manifest['model_loading_info']['strict_complete'] and manifest['longest_preflight']['passed']
    assert manifest['model_loading_info']['protein_stage'] == 'inference'
    records = manifest['records']
    assert len(items) == len(records) == 13192 and set(items) == set(records)
    pooled = {}
    max_reference_difference = 0.0
    torch.set_num_threads(1)
    for index,(key,item) in enumerate(items.items()):
        rec = records[key]
        assert hashlib.sha256(item['input'].encode()).hexdigest() == key
        assert rec['file'] == key+'.npy' and rec['shape'] == [len(item['input']),1024]
        content = (cache/rec['file']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == rec['sha256'],key
        array = np.load(io.BytesIO(content),allow_pickle=False)
        assert list(array.shape) == rec['shape'] and rec['dtype'] == 'float32'
        vector = pool(array)
        assert vector.shape == (1024,) and vector.dtype == np.float32 and np.isfinite(vector).all()
        if index in (0,len(items)//2,len(items)-1):
            reference = np.mean(array.astype(np.float64),axis=0)
            error = float(np.max(np.abs(vector-reference)))
            assert np.allclose(vector,reference,rtol=1e-5,atol=1e-6)
            max_reference_difference = max(error,max_reference_difference)
        pooled[key] = vector
        if (index+1)%1000 == 0:
            save(ROOT/'prepare_progress.json',dict(verified_and_pooled=index+1,total=len(items)))
            print(json.dumps(dict(verified_and_pooled=index+1,total=len(items))),flush=True)
    old = np.load(BASE/'inputs/feature1959.npy',allow_pickle=False,mmap_mode='r')
    frame = pd.read_csv(BASE/'inputs/rows.csv',dtype={'author_row_id':str},keep_default_na=False,float_precision='round_trip')
    assert old.shape == (27658,1959) and old.dtype == np.float32 and len(frame) == 27658
    assert frame.author_row_id.is_unique
    new = np.array(old,copy=True)
    keys = []
    for i,row in enumerate(frame.itertuples(index=False)):
        key = hashlib.sha256(protein_input(row.sequence).encode()).hexdigest()
        assert key in pooled
        new[i,:1024] = pooled[key]
        keys.append(key)
    assert set(keys) == set(pooled) and np.isfinite(new).all()
    assert np.array_equal(new[:,1024:],old[:,1024:])
    assert not np.array_equal(new[:,:1024],old[:,:1024])
    labels = frame.log10_kcat.to_numpy(float)
    assert np.allclose(labels,np.log10(frame.kcat_s_minus1.to_numpy(float)),atol=1e-12,rtol=0)
    np.save(output,new,allow_pickle=False)
    oldmeta = read(BASE/'inputs/feature_store.json')
    feature_meta = dict(oldmeta,rows_csv=str(BASE/'inputs/rows.csv'),features_npy='feature1959.npy',
        features_npy_sha256=sha(output),feature_blocks=[dict(name='RAMER',width=1024),dict(name='MolT5',width=768),dict(name='MACCS',width=167)],
        replacement=dict(cache_manifest_sha256=CACHE_SHA,protein_items_sha256=ITEMS_SHA,
            pooling='sum over valid cached residues / valid residue count; CPU float32 torch; no padding or EOS',
            protein_preprocessing='original CataPro >1000 first500+last500; UZOB->X',
            original_feature_sha256=EXPECTED['inputs/feature1959.npy'],original_source_hashes=oldmeta['source_hashes']))
    feature_meta.pop('source_hashes')
    save(ROOT/'inputs/feature_store.json',feature_meta)
    report = dict(status='passed',rows=27658,unique_proteins=len(pooled),all_cache_sha_verified=True,
        unchanged_substrate_columns=935,substrate_exact=True,labels_unchanged=True,
        pooled_reference_max_abs_difference=max_reference_difference,feature_sha256=sha(output),
        feature_manifest_sha256=sha(ROOT/'inputs/feature_store.json'),cache_manifest_sha256=CACHE_SHA,
        old_input_sha256=EXPECTED)
    save(ROOT/'prepare_result.json',report)
    print(json.dumps(report),flush=True)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def review(device):
    sources=source_check()
    assert read(ROOT/'prepare_result.json')['status'] == 'passed'
    if device.startswith('cuda'): assert torch.cuda.is_available()
    torch.set_num_threads(1)
    runner=module('ramer_original_runner',ROOT/'catapro_baseline/runner.py')
    author=module('ramer_author_model',ROOT/'author/training/kcat/model.py')
    utils=module('ramer_author_utils',ROOT/'author/training/kcat/utils.py')
    store=MasterStore(ROOT/'inputs/feature_store.json')
    original=np.load(BASE/'inputs/feature1959.npy',allow_pickle=False,mmap_mode='r')
    assert np.array_equal(store.features[:,1024:],original[:,1024:])
    spec=read(ROOT/'inputs/rowspec.json')
    assignment=pd.read_csv(ROOT/'assignments.csv')
    membership=dict(zip(assignment.balance_row_uid.str.removeprefix('catapro:'),assignment.new_split))
    all_ids=[]
    for split in COUNTS:
        pairs=spec['eval']['test'] if split == 'test' else spec[split]
        assert len(pairs) == COUNTS[split]
        for source,uid in pairs:
            assert source == 'catapro' and membership[uid] == split
            all_ids.append(uid)
    assert len(set(all_ids)) == len(store) == 27658 and set(all_ids) == set(store.indices)
    # Deterministic batch coverage is independent of DataLoader worker scheduling.
    sampler=MergeSingletonBatchSampler(list(range(COUNTS['train'])),batch_size=8,drop_last=False)
    batches=list(sampler)
    assert len(batches[-1]) == 9 and [i for b in batches for i in b] == list(range(COUNTS['train']))
    check_array=np.arange(3*1024,dtype=np.float32).reshape(3,1024)
    assert np.array_equal(pool(check_array),check_array.mean(axis=0))
    train_ids=[uid for _,uid in spec['train']]
    selected=[train_ids[:8],train_ids[-9:]]
    data=[]
    for ids in selected:
        rows=[store.get('catapro',uid) for uid in ids]
        data.append(torch.tensor(np.stack([np.r_[r['features'],np.float32(r['label'])] for r in rows]),dtype=torch.float32))
    runner.set_seed(668)
    m1=author.KcatModel(0.0,device=device)
    initial=state_sha(m1)
    # Actual old author model source must be identical, then reconstruct its initial state.
    old_path=BASE/'catapro_baseline/author/training/kcat/model.py'
    assert sha(old_path) == sha(ROOT/'author/training/kcat/model.py')
    old_author=module('old_catapro_model_reference',old_path)
    runner.set_seed(668)
    m2=old_author.KcatModel(0.0,device=device)
    assert initial == state_sha(m2)
    opts=[torch.optim.Adam(m.parameters(),lr=1e-5,betas=(.9,.999),weight_decay=.01) for m in (m1,m2)]
    before={n:int(v) for n,v in m1.state_dict().items() if n.endswith('num_batches_tracked')}
    values=[utils.run_a_training_epoch(m,data,opt,device) for m,opt in zip((m1,m2),opts)]
    assert np.array_equal(values[0],values[1]) and np.isfinite(values).all()
    assert state_sha(m1) == state_sha(m2)
    for n,v in m1.state_dict().items():
        assert torch.isfinite(v).all(),n
        if n in before: assert int(v)-before[n] == 2,n
    gradients={n:float(p.grad.detach().norm()) for n,p in m1.named_parameters() if p.grad is not None}
    assert len(gradients) == len(list(m1.parameters())) and all(np.isfinite(v) for v in gradients.values())
    assert any(v > 0 for v in gradients.values())
    b=data[0].to(device)
    m1.eval()
    with torch.no_grad(): expected=m1(b[:,:1024],b[:,1024:-1])
    memory=io.BytesIO()
    torch.save(dict(model=m1.state_dict(),optimizer=opts[0].state_dict()),memory)
    memory.seek(0)
    payload=torch.load(memory,map_location=device,weights_only=False)
    m2.load_state_dict(payload['model'],strict=True);opts[1].load_state_dict(payload['optimizer']);m2.eval()
    with torch.no_grad(): actual=m2(b[:,:1024],b[:,1024:-1])
    assert torch.equal(actual,expected)
    # One further identical step checks the Adam moments were really restored.
    for m,opt in zip((m1,m2),opts): utils.run_a_training_epoch(m,[data[0]],opt,device)
    assert state_sha(m1) == state_sha(m2)
    stopper=utils.EarlyStopping(20,.001)
    flags=[stopper.check(i,v) for i,v in enumerate([1.5,1.4995,1.498]+[1.499]*20)]
    assert flags[0] == (True,False) and flags[1] == (False,False) and flags[2] == (True,False)
    assert flags[-1] == (False,True)
    result=dict(status='passed',device=device,source_check=sources,rows=COUNTS,
        parameters=sum(p.numel() for p in m1.parameters()),initial_state_sha256=initial,
        original_model_initial_exact=True,original_two_batch_updates_exact=True,
        bn_updates_per_module=2,gradient_finite=True,model_adam_roundtrip_exact=True,
        train_coverage_exact=True,tail_batch_size=9,selection_rule_checked=True,
        pooled_mask_reference_checked=True,substrate_exact=True,test_forward=False,
        discarded_review_updates=True,torch_version=torch.__version__,numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0) if device.startswith('cuda') else None)
    save(ROOT/('review_cuda.json' if device.startswith('cuda') else 'review_cpu.json'),result)
    print(json.dumps(result),flush=True)


def audit_result():
    source_check()
    run=ROOT/'run_seed668'
    protocol=read(run/'protocol.json');summary=read(run/'summary.json');history=read(run/'history.json')
    ref=read(ROOT/'reference_protocol.json')
    assert protocol['parameters'] == ref['parameters']
    assert protocol['source_sha256'] == ref['source_sha256']
    assert summary['architecture'] == ref['architecture']
    anchor=None;counter=0;best=None
    for i,row in enumerate(history):
        loss=row['valid']['batch_average_RMSELoss'];improved=i==0 or loss<anchor-.001
        if improved:anchor=loss;best=i;counter=1 if i==0 else 0
        else:counter+=1
        assert row['epoch']==i and row['is_best']==improved and row['early_stopping']==(counter==20)
        assert counter!=20 or i==len(history)-1
    assert counter==20 or len(history)==150
    assert summary['best_epoch_zero_based']==best and summary['epochs_ran']==len(history)
    assert not summary['strict_load_missing_keys'] and not summary['strict_load_unexpected_keys']
    store=MasterStore(ROOT/'inputs/feature_store.json');spec=read(ROOT/'inputs/rowspec.json')
    metrics={}
    for split in ['valid','test']:
        frame=pd.read_csv(run/(split+'_predictions.csv'),dtype={'row_uid':str},float_precision='round_trip')
        ids=[uid for _,uid in (spec['eval']['test'] if split=='test' else spec['valid'])]
        assert frame.row_uid.tolist()==ids and frame.row_uid.is_unique and len(frame)==COUNTS[split]
        y=np.asarray([store.get('catapro',uid)['label'] for uid in ids],dtype=np.float32)
        assert np.array_equal(y,frame.log_Kcat.to_numpy(np.float32))
        pred=frame.prediction.to_numpy(float);y=y.astype(float)
        assert np.isfinite(pred).all()
        mse=np.mean((pred-y)**2)
        m=dict(RMSE=float(np.sqrt(mse)),MAE=float(np.mean(abs(pred-y))),R2=float(1-mse/np.var(y)),PCC=float(np.corrcoef(y,pred)[0,1]))
        assert all(abs(m[k]-summary['results'][split]['metrics'][k])<3e-6 for k in m)
        batch=float(np.mean([np.sqrt(np.mean((pred[j:j+8]-y[j:j+8])**2)+1e-6) for j in range(0,len(y),8)]))
        assert abs(batch-summary['results'][split]['reported_batch_average_RMSELoss'])<3e-6
        if split=='valid':
            assert abs(batch-history[best]['valid']['batch_average_RMSELoss'])<3e-6
            assert all(abs(m[k]-history[best]['valid'][k])<3e-6 for k in ['RMSE','R2','PCC'])
        metrics[split]=m
    result=dict(status='passed',best_epoch=best+1,completed_epochs=len(history),metrics=metrics,
        selected_valid_metrics_replayed=True,test_rows=3036,checkpoint_sha256=sha(run/'checkpoints/bestmodel.pth'),
        prediction_sha256={s:sha(run/(s+'_predictions.csv')) for s in ['valid','test']},
        scope='Original baseline evaluation procedure; valid metrics checked against selected history; no saved per-value training valid reference in original runner')
    save(ROOT/'result_audit.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','review_cpu','review_cuda','audit_result'])
    args=parser.parse_args()
    try:
        save(ROOT/'pipeline_status.json',dict(stage=args.stage,state='running',utc=datetime.now(timezone.utc).isoformat()))
        {'prepare':prepare,'review_cpu':lambda:review('cpu'),'review_cuda':lambda:review('cuda:0'),'audit_result':audit_result}[args.stage]()
        save(ROOT/'pipeline_status.json',dict(stage=args.stage,state='complete',utc=datetime.now(timezone.utc).isoformat()))
    except Exception as error:
        save(ROOT/'pipeline_status.json',dict(stage=args.stage,state='failed',error=repr(error),traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        raise
