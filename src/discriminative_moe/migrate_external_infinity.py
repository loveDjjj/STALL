"""仅修复JSON Infinity读取的可审计恢复；保持全部已存特征数组逐元素不变。"""
import hashlib,json
from pathlib import Path
import numpy as np
from artifacts import paper_json
from reference import file_digest
from config import config_digest
from discriminative_moe.run import ROOT,configuration


def migrate(root=ROOT):
    root=Path(root);out=root/configuration(root)['run_directory']/'external'
    receipt=out/'infinity_parser_recovery.json'
    if receipt.exists():raise ValueError('此一次性恢复已经执行，不重复迁移')
    prepared=json.loads((out/'prepared.json').read_text());old=json.loads((out/'features/identity.json').read_text())
    path=root/'src/discriminative_moe/external.py';text=path.read_text();line='                b=np.asarray(b,dtype=np.float64)\n'
    assert text.count(line)==1
    original=hashlib.sha256(text.replace(line,'').encode()).hexdigest()
    assert original==old['code']['src/discriminative_moe/external.py']==prepared['identity']['code']
    for p,h in prepared['identity']['inputs'].items():assert file_digest(root/p)==h
    for p,h in prepared['files'].items():assert file_digest(out/p)==h
    old_id=config_digest(old);prepared['identity']['code']=file_digest(path)
    paper_json(out/'prepared.json',prepared)
    new=json.loads(json.dumps(old));new['code']['src/discriminative_moe/external.py']=file_digest(path);new['prepared']=file_digest(out/'prepared.json');new_id=config_digest(new)
    rows=[]
    for p in sorted((out/'features').glob('*.npz')):
        old_hash=file_digest(p)
        with np.load(p) as z:
            assert str(z['identity'])==old_id
            arrays={k:z[k].copy() for k in z.files}
        originals={k:arrays[k].copy() for k in ['G','T','L']}
        arrays['identity']=np.asarray(new_id);tmp=p.with_suffix('.tmp.npz');np.savez(tmp,**arrays)
        with np.load(tmp) as z:
            for k,a in originals.items():np.testing.assert_array_equal(z[k],a)
        tmp.replace(p);rows.append(dict(file=p.name,previous=old_hash,current=file_digest(p)))
    paper_json(out/'features/identity.json',new)
    paper_json(receipt,dict(status='verified',old_identity=old_id,new_identity=new_id,
        only_change='Cast serialized reference Infinity back to float64 before comparison; encoder, inputs and feature mathematics unchanged',
        source_before=original,source_after=file_digest(path),arrays_exactly_preserved=True,files=rows))
    print('Recovered',len(rows),'unchanged feature arrays',flush=True)


if __name__=='__main__':migrate()
