"""Export the exact frozen row membership/order; never generate a new split."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/catapro_baseline_ramer_cluster40_20261029'
SOURCE_SHA='a0f15c13c715006513040387a5de600fdffc42481af5830277e683d5a6dc1710'
COUNTS={'train':21585,'valid':3037,'test':3036}

def sha(data):return hashlib.sha256(data).hexdigest()

def export(source,destination):
    raw=source.read_bytes()
    assert sha(raw)==SOURCE_SHA,'Source must be the exact table used by the completed experiment'
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8'),newline=''))
    fields=reader.fieldnames
    assert fields==['author_row_id','fold','sequence','smiles','kcat_s_minus1','log10_kcat']
    rows=list(reader);by_id={r['author_row_id']:r for r in rows}
    assert len(rows)==len(by_id)==sum(COUNTS.values())
    spec_path=EXPERIMENT/'inputs/rowspec.json';assignment_path=EXPERIMENT/'assignments.csv'
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    assert sha(assignment_path.read_bytes())==spec['meta']['assignments_sha256']
    assignment=list(csv.DictReader(assignment_path.open(encoding='utf-8',newline='')))
    membership={r['balance_row_uid'].removeprefix('catapro:'):r for r in assignment}
    assert len(membership)==len(assignment)==len(rows) and set(membership)==set(by_id)
    destination.mkdir(parents=True,exist_ok=True)
    outputs={};all_ids=[];clusters={}
    for split,count in COUNTS.items():
        pairs=spec['eval']['test'] if split=='test' else spec[split]
        assert len(pairs)==count
        buf=io.StringIO(newline='')
        writer=csv.DictWriter(buf,fieldnames=fields+['split','cluster_id'],lineterminator='\n')
        writer.writeheader();ids=[];groups=set()
        for src,uid in pairs:
            assert src=='catapro' and membership[uid]['new_split']==split
            original=by_id[uid]
            kcat=float(original['kcat_s_minus1']);label=float(original['log10_kcat'])
            assert kcat>0 and math.isfinite(label) and abs(math.log10(kcat)-label)<1e-12
            group=membership[uid]['balance_group']
            writer.writerow(dict(original,split=split,cluster_id=group))
            ids.append(uid);groups.add(group)
        data=buf.getvalue().encode('utf-8');path=destination/(split+'.csv')
        if path.exists():assert path.read_bytes()==data,'Refusing to overwrite a different export'
        else:path.write_bytes(data)
        # Re-read actual CSV output and compare all original field strings, including labels.
        exported=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
        assert [r['author_row_id'] for r in exported]==ids
        for row in exported:
            assert {k:row[k] for k in fields}==by_id[row['author_row_id']]
        outputs[split]=dict(file=path.name,rows=count,bytes=len(data),sha256=sha(data),clusters=len(groups),
            first_row_id=ids[0],last_row_id=ids[-1],ordered_ids_sha256=sha(('\n'.join(ids)+'\n').encode()))
        clusters[split]=groups;all_ids+=ids
    assert len(set(all_ids))==len(all_ids) and set(all_ids)==set(by_id)
    for a,b in [('train','valid'),('train','test'),('valid','test')]:assert not clusters[a]&clusters[b],(a,b)
    manifest=dict(dataset='CataPro original rows, fixed cluster40 seed668',source_rows_sha256=SOURCE_SHA,
        rowspec_sha256=sha(spec_path.read_bytes()),assignments_sha256=sha(assignment_path.read_bytes()),
        rows=27658,columns=fields+['split','cluster_id'],files=outputs,
        all_original_field_strings_preserved=True,order_matches_rowspec=True,row_overlap=False,cluster_overlap=False,
        note='fold is the original author annotation and is NOT the current split; sequence remains full length; embeddings are external')
    content=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    path=destination/'manifest.json'
    if path.exists():assert path.read_bytes()==content
    else:path.write_bytes(content)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/cluster40_seed668')
    args=parser.parse_args();export(args.rows,args.output)
