"""特征身份与源级用途：先划分再训练，不将配对表展开为训练样本。"""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from artifacts import atomic_csv, paper_json, checkpoint_read
from config import config_digest
from reference import file_digest
from discriminative_moe.run import configuration
from discriminative_moe.features import projection, global_descriptors


def grouped_roles(frame, seed=17, fraction=.2):
    """每个连接源组只归一个用途；跨域同源在相应fold中禁止进入训练。"""
    role={}
    groups=[]
    for group,q in frame.groupby('split_group',sort=True):
        if q.subset.nunique()!=1:
            # 联合内容源可以跨真假；源stratum仅决定抽样，不改变其不可拆分性。
            stratum='mixed:'+','.join(sorted(q.dataset.unique()))
        else:
            stratum='|'.join(sorted(set(q.dataset+'|'+q.subset+'|'+q.source_model)))
        groups.append((group,stratum))
    for stratum in sorted({s for _,s in groups}):
        members=[g for g,s in groups if s==stratum]
        salt=int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:4],'little')
        rng=np.random.default_rng(np.random.SeedSequence([seed,salt]));order=rng.permutation(len(members))
        n=max(1,round(fraction*len(members))) if len(members)>1 else 0
        chosen={members[i] for i in order[:n]}
        role.update({g:('validation' if g in chosen else 'train') for g in members})
    rows=[]
    domains=sorted(frame.dataset.unique())
    for fold in domains+['pooled']:
        blocked=set(frame.loc[frame.dataset.eq(fold),'split_group']) if fold!='pooled' else set()
        for q in frame.itertuples():
            use='test' if q.dataset==fold else ('excluded_shared_source' if q.split_group in blocked else role[q.split_group])
            rows.append(dict(fold=fold,video_id=q.video_id,split_group=q.split_group,dataset=q.dataset,subset=q.subset,source_model=q.source_model,role=use))
    return pd.DataFrame(rows)


def connected_groups(frame, root):
    """合并已有源组与同物理文件；不声称完成语义去重。"""
    parent=list(range(len(frame)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(frame.itertuples()):
        stat=(root/r.video_path).stat()
        keys=[('source',r.source_group),('inode',stat.st_dev,stat.st_ino)]
        if r.content_sha256:keys.append(('sha256',r.content_sha256))
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    names={}
    for i,r in enumerate(frame.itertuples()):names.setdefault(find(i),[]).append(r.video_id)
    mapping={i:'source:'+hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest()[:24] for i,v in names.items()}
    return [mapping[find(i)] for i in range(len(frame))]


def prepare(root):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];source=root/c['evidence_directory']
    spec=dict(config=c,inputs={p:file_digest(root/p) for p in ['results/paper_complete/evaluation.csv','results/paper_complete/pairs.csv',
        c['evidence_directory']+'/plans.json',c['evidence_directory']+'/dense/identity.json']},
        code={p:file_digest(root/p) for p in ['src/discriminative_moe/data.py','src/discriminative_moe/features.py']})
    marker=out/'prepared.json'
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['identity']!=config_digest(spec):raise ValueError('准备身份改变')
        for p,h in m['files'].items():
            if file_digest(out/p)!=h:raise ValueError('准备产物改变：'+p)
        print('supervised data already verified',flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(root/'results/paper_complete/evaluation.csv',keep_default_na=False)
    if meta.video_id.duplicated().any():raise ValueError('clip身份重复')
    meta['split_group']=connected_groups(meta,root)
    oldspec=json.loads((source/'dense/identity.json').read_text());oldid=config_digest(oldspec)
    basis=projection(rank=c['projection_dimension'],seed=c['projection_seed'])
    jobs=[j for j in json.loads((source/'plans.json').read_text())['jobs'] if j['role']=='evaluation']
    lookup=meta.set_index('video_id');rows=[];gall=[];tall=[];kg=[]
    for i,j in enumerate(jobs):
        payload=checkpoint_read(source/'dense'/(j['key']+'.json'),oldid,j['key'])
        gp=source/'global'/(j['key']+'.npz')
        if file_digest(gp)!=payload['global_sha256']:raise ValueError('旧Global缓存损坏')
        with np.load(gp) as z:
            if str(z['identity'])!=oldid or str(z['key'])!=j['key']:raise ValueError('Global合同不一致')
            g,t=global_descriptors(z['global_features'],basis)
        for domain,m in j['targets'].items():
            r=lookup.loc[m['video_id']]
            if (r.dataset,r.source_group,r.subset)!=(domain,m['source_group'],m['subset']):raise ValueError('身份漂移')
            rows.append(dict(index=len(rows),video_id=m['video_id'],video_path=j['video_path'],dataset=domain,subset=r.subset,
                source_model=r.source_model,source_group=r.source_group,split_group=r.split_group,real_source=r.real_source,
                key=j['key'],global_sha256=payload['global_sha256'],effective_k=len(g),length=len(j['windows'][0])))
            gpadded=np.zeros((3,g.shape[-1]),np.float32);tpadded=np.zeros((3,t.shape[-1]),np.float32)
            gpadded[:len(g)]=g;tpadded[:len(t)]=t;gall.append(gpadded);tall.append(tpadded);kg.append(len(g))
        if (i+1)%2000==0:print('Global descriptors',i+1,len(jobs),flush=True)
    frame=pd.DataFrame(rows)
    if set(frame.video_id)!=set(meta.video_id) or len(frame)!=15569:raise ValueError('覆盖不完整')
    roles=grouped_roles(frame,c['split_seed'],c['validation_fraction'])
    for fold,q in roles.groupby('fold'):
        sets={r:set(z.split_group) for r,z in q[q.role!='excluded_shared_source'].groupby('role')}
        for a in sets:
            for b in sets:
                if a!=b and sets[a]&sets[b]:raise ValueError('源级用途泄漏')
    atomic_csv(out/'videos.csv',frame);atomic_csv(out/'roles.csv',roles)
    atomic_csv(out/'split_counts.csv',roles.groupby(['fold','role','dataset','subset','source_model']).agg(clips=('video_id','size'),sources=('split_group','nunique')).reset_index())
    np.savez(out/'global_features.npz',G=np.stack(gall),T=np.stack(tall),K=np.asarray(kg),projection=basis.numpy(),video_ids=frame.video_id.to_numpy(dtype=str))
    paper_json(out/'jobs.json',dict(jobs=jobs))
    files=['videos.csv','roles.csv','split_counts.csv','global_features.npz','jobs.json']
    paper_json(marker,dict(status='prepared',identity=config_digest(spec),spec=spec,clips=len(frame),
        source_groups=frame.split_group.nunique(),files={p:file_digest(out/p) for p in files}))
    print('supervised preparation complete',len(frame),frame.split_group.nunique(),flush=True)
