"""固定原窗口与干净参考的H.264重编码对照；原视频不改动，不缓存新Patch。"""
import argparse,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from data.prefetch import bounded_map,frame_reservation
from data.video import decode_bounded
from evaluation.confirmation_data import context,DOMAINS
from evaluation.confirmation_engine import Chunks
from evaluation.confirmation_run import wait_pid
from evaluation.direction_features import direction_features
from evaluation.direction_study import parameter
from evaluation.direction_analysis import shared_contrasts
from evaluation.tables import evaluate_fixed_pairs
from features import AlphaStallFeatureExtractor
from branches.global_branch import score_global_raw
from math_utils import GaussianMeanCandidateScorerFloat64,l2_normalized_second_order
from reference import file_digest,local_video_cdfs,percentile,window_mean


def timestamps(path,with_shape=False):
    command=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=width,height','-of','json',str(path)]
    response=subprocess.run(command,check=True,capture_output=True,text=True,timeout=180)
    payload=json.loads(response.stdout);frames=payload['frames'];times=np.array([float(f['best_effort_timestamp_time']) for f in frames])
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<0):raise ValueError('视频时间戳无效')
    times=times-times[0]
    shape=(int(payload['streams'][0]['width']),int(payload['streams'][0]['height']))
    return (times,shape) if with_shape else times


def transcode(source,destination,crf,c,input_timebase=False):
    """4:4:4保持奇数边长，不偷偷补边、缩放或复制帧。"""
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-copyts','-start_at_zero','-i',str(source),
        '-map','0:v:0','-an','-sn','-dn','-c:v',c['codec'],'-preset',c['preset'],'-crf',str(crf),
        '-pix_fmt',c['pixel_format'],'-threads',str(c['encoder_threads']),'-vsync','0']
    if input_timebase:command+=['-enc_time_base','-1']
    command.append(str(destination))
    subprocess.run(command,check=True,capture_output=True,text=True,timeout=600)
    return command


def setup(root):
    c,parent,data=context(root);out=parent/'encoding';out.mkdir(exist_ok=True)
    e=yaml.safe_load((root/'configs/paper.yaml').read_text())['encoder']
    paths=[parent/'robust_jobs.json',parent/'robust_pairs.csv',root/'results/runs/local_direction_evidence/cdf_arrays.npz']
    paths += [root/f'results/runs/paper_fit_{d}/gaussians.npz' for d in DOMAINS]
    spec=dict(config=c,encoder=e,inputs={str(p.relative_to(root)):file_digest(p) for p in paths},code=file_digest(Path(__file__)),
        dependencies={p:file_digest(root/p) for p in ('src/features.py','src/math_utils.py','src/branches/global_branch.py','src/evaluation/direction_features.py','src/data/video.py')},
        policy='fixed original FC windows; clean references; yuv444p H264 encoding incl pixel-format conversion; no resampling')
    marker=out/'identity.json'
    if marker.exists() and json.loads(marker.read_text())!=spec:
        prior=json.loads(marker.read_text());p=out/'timebase_compatibility.json'
        proof=json.loads(p.read_text()) if p.exists() else {}
        if proof.get('original_spec')!=prior or proof.get('current_spec')!=spec or proof.get('status')!='validated':
            raise ValueError('重编码协议改变且缺少已验证兼容记录')
        # 已通过时间戳合同的旧输出不重编码；新增记录另存实际生产代码和时间基选择。
        return c,parent,out,e,config_digest(prior)
    if not marker.exists():paper_json(marker,spec)
    return c,parent,out,e,config_digest(spec)


def jobs_by_path(parent):
    groups={}
    for j in json.loads((parent/'robust_jobs.json').read_text())['jobs']:groups.setdefault(j['video_path'],[]).append(j)
    return [dict(path=p,views=v,key=config_digest(dict(path=p,views=[j['key'] for j in v]))) for p,v in groups.items()]


def worker(root,rank=0,world=2,probe=False):
    c,parent,out,e,identity=setup(root);items=jobs_by_path(parent)[rank::world]
    if probe:
        seen=set();chosen=[]
        for item in items:
            j=item['views'][0];d=next(iter(j['targets']));tag=(d,j['targets'][d]['subset'],len(j['windows'][0]))
            if tag not in seen:chosen.append(item);seen.add(tag)
        items=chosen
    directory=out/('probe' if probe else 'raw')/f'rank{rank}';store=Chunks(directory,identity)
    pending=[(item,crf) for item in items for crf in c['crf'] if item['key']+f'_crf{crf}' not in store.records]
    if not pending:return
    if file_digest(root/e['weights'])!=e['weights_sha256']:raise ValueError('DINO权重改变')
    extractor=AlphaStallFeatureExtractor(c['devices'][rank],dino_repo=str(root/e['repo']),dino_weights=str(root/e['weights']),pad_tail_batch=True)
    center=parameter(root/'results/runs/paper_fit_source/gaussians.npz','lt').mean;models={}
    for d in DOMAINS:
        p=root/f'results/runs/paper_fit_{d}/gaussians.npz'
        models[d]=(parameter(p,'gs'),parameter(p,'gt'),GaussianMeanCandidateScorerFloat64([parameter(p,'lt')],center,c['devices'][rank]))
    producer_code=file_digest(Path(__file__))
    compatibility=out/'timebase_compatibility.json'
    known=set(json.loads(compatibility.read_text()).get('default_timebase_failed_paths',[])) if compatibility.exists() else set()
    def prepare(entry):
        item,crf=entry;source=root/item['path'];j=item['views'][0];st=source.stat()
        if (st.st_size,st.st_mtime_ns)!=(j['source_bytes'],j['source_mtime_ns']):raise ValueError('重编码前源视频改变')
        tmp=tempfile.TemporaryDirectory(prefix='encoding-',dir=out);destination=Path(tmp.name)/'video.mp4';start=time.perf_counter()
        try:
            before,shape=timestamps(source,True);preserve=item['path'] in known
            command=transcode(source,destination,crf,c,input_timebase=preserve);after,encoded_shape=timestamps(destination,True)
            if len(before)!=len(after):raise ValueError('重编码改变帧数')
            if shape!=encoded_shape:raise ValueError('重编码改变图像尺寸')
            error=float(np.max(np.abs(before-after)))
            initial_error=error
            if error>1e-3 and not preserve:
                command=transcode(source,destination,crf,c,input_timebase=True);after,encoded_shape=timestamps(destination,True);preserve=True
                if len(before)!=len(after) or shape!=encoded_shape:raise ValueError('保留输入时间基后帧数/尺寸改变')
                error=float(np.max(np.abs(before-after)))
            if error>1e-3:raise ValueError(f'保留时间基后仍超过1ms：{item["path"]}, crf={crf}, error={error}')
            prepared=[]
            for view in item['views']:
                union=sorted({x for w in view['windows'] for x in w})
                frames=extractor.prepare_frames(decode_bounded(destination,union))
                clean=extractor.prepare_frames(decode_bounded(source,union)) if probe else None
                prepared.append((view,union,frames,clean))
            receipt=dict(command=command[:-1]+['<temporary-output>'],frames=len(before),shape=list(shape),max_timestamp_error=error,
                encoded_bytes=destination.stat().st_size,source_bytes=st.st_size,seconds=time.perf_counter()-start,
                timestamp_sha256=config_digest(before.tolist()),encoder_timebase='input' if preserve else 'default',
                initial_timestamp_error=initial_error,producer_code_sha256=producer_code)
            return prepared,receipt,tmp
        except BaseException as exc:
            paper_json(out/'failures'/(item['key']+f'_crf{crf}.json'),dict(video_path=item['path'],crf=crf,error=str(exc),producer_code=producer_code))
            tmp.cleanup();raise
    def reserve(entry):
        item,crf=entry
        return max(frame_reservation(root/item['path'],len({x for w in j['windows'] for x in w})) for j in item['views'])+128*2**20
    stream=bounded_map(pending,prepare,reserve,workers=2,depth=2,budget=12*2**30);start=time.perf_counter()
    try:
        for n,((item,crf),(views,receipt,tmp)) in enumerate(stream,1):
            records=[]
            try:
                for j,union,frames,clean in views:
                    where={x:i for i,x in enumerate(union)};pick=[[where[x] for x in w] for w in j['windows']]
                    def infer(input_frames):
                        f=extractor.frames_to_global_patch_embeddings([input_frames],batch_size=8)[0]
                        g=np.stack([f['global'][ix] for ix in pick]);p=np.stack([f['patch'][ix] for ix in pick]);u=l2_normalized_second_order(torch.from_numpy(p));_,scalar=direction_features(p)
                        values={}
                        for d,meta in j['targets'].items():
                            gs,gt,scorer=models[d];global_raw=score_global_raw(g,gs,gt,device=c['devices'][rank]);local=scorer.score(u)[:,0]
                            values[d]=dict(gs=global_raw.spatial.tolist(),gt=global_raw.temporal_t1.tolist(),local=local.tolist())
                        return values,scalar.tolist()
                    actual,scalar=infer(frames)
                    if clean is not None:
                        original,_=infer(clean)
                        for d,meta in j['targets'].items():
                            expected=np.array([[float(w[k]) for k in ('gs','gt','lt')] for w in meta['expected']])
                            obs=np.stack([original[d]['gs'],original[d]['gt'],original[d]['local']],axis=1)
                            np.testing.assert_allclose(obs,expected,rtol=0,atol=1e-8)
                    records.append(dict(key=j['key'],targets=actual,scalars=scalar))
                store.add(dict(key=item['key']+f'_crf{crf}',crf=crf,views=records,receipt=receipt))
            finally:tmp.cleanup()
            if n%8==0 or n==len(pending):
                store.flush();elapsed=time.perf_counter()-start
                paper_json(out/f'progress_{rank}.json',dict(completed=n,total=len(pending),probe=probe,seconds=elapsed,eta_seconds=(len(pending)-n)*elapsed/n))
                print('encoding',rank,n,len(pending),'probe' if probe else 'full',flush=True)
    finally:stream.close()
    store.flush();paper_json(out/f'{"probe" if probe else "done"}_{rank}.json',dict(status='completed',identity=identity,probe=probe))


def evaluate(root):
    c,parent,out,e,identity=setup(root);records={}
    for rank in (0,1):
        store=Chunks(out/'raw'/f'rank{rank}',identity)
        for item in store.records.values():
            for j in item['views']:
                key=(item['crf'],j['key'])
                if key in records:raise ValueError('重编码视图重复')
                records[key]=j
    selected=json.loads((parent/'robust_jobs.json').read_text())['jobs'];all_jobs=json.loads((root/c['plans']).read_text())['jobs'];gc={};lc={}
    for j in all_jobs:
        if j['role']!='cdf':continue
        t=len(j['windows'][0])
        for d,m in j['targets'].items():
            for branch in ('gs','gt'):gc.setdefault((d,t,branch),[]).append(float(m['uniform'][branch]))
            lc.setdefault((d,t),[]).append([float(w['lt']) for w in m['expected']])
    lc={k:local_video_cdfs(v) if k[1]==16 else {1:np.sort(np.asarray(v)[:,0])} for k,v in lc.items()}
    scalar_refs=np.load(root/'results/runs/local_direction_evidence/cdf_arrays.npz');rows=[]
    for crf in c['crf']:
        for j in selected:
            record=records[crf,j['key']];t=len(j['windows'][0]);k=len(j['windows'])
            for d,meta in j['targets'].items():
                raw=record['targets'][d];gs=percentile([float(x) for x in raw['gs']],gc[d,t,'gs']);gt=percentile([float(x) for x in raw['gt']],gc[d,t,'gt'],allow_positive_infinity=True)
                g=window_mean(.5*gs+.5*gt);q=window_mean(raw['local']);l=float(percentile([q],lc[d,t][k])[0]);v=dict(global_only=g,full=.5*g+.5*l,local_raw=q)
                for i,name in ((0,'ttr'),(2,'split')):
                    r=window_mean(-np.asarray(record['scalars'])[:,i]);score=float(percentile([r],scalar_refs[f'{d}__T{t}__{name}__K{k}'])[0]);v[name]=.5*g+.5*score;v[name+'_raw']=r
                fields={f:meta[f] for f in ('video_id','dataset','subset','source_model','source_group')};fields.update(window_frames=t)
                rows.extend([dict(**fields,condition=f'crf{crf}',variant=name,final_score=value) for name,value in v.items()])
    wanted={m['video_id'] for j in selected for m in j['targets'].values()};meta=pd.read_csv(parent/'robust_evaluation.csv',keep_default_na=False).set_index('video_id')
    aliases={'global_only':'global_only','d2_anchor':'full','d2_anchor_raw':'local_raw','ttr':'ttr','split':'split','ttr_raw':'ttr_raw','split_raw':'split_raw'}
    base=pd.read_csv(root/'results/runs/local_direction_evidence/video_scores.csv.gz',float_precision='round_trip')
    for r in base[base.video_id.isin(wanted)&base.variant.isin(aliases)].itertuples():
        rows.append(dict(video_id=r.video_id,dataset=r.dataset,subset=r.subset,source_model=r.source_model,source_group=meta.loc[r.video_id,'source_group'],window_frames=r.window_frames,condition='clean',variant=aliases[r.variant],final_score=r.final_score))
    scores=pd.DataFrame(rows);pairs=pd.read_csv(parent/'robust_pairs.csv');tables={}
    for (cond,variant),part in scores.groupby(['condition','variant']):
        if set(part.video_id)!=wanted:raise ValueError('条件间评价身份不完整')
        for key,t in evaluate_fixed_pairs(part,pairs).items():tables.setdefault(key,[]).append(t.assign(condition=cond,variant=variant))
    scores.to_csv(out/'video_scores.csv.gz',index=False)
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t).replace({'scope':{'Macro-3':'Average'}}))
    frames={f'{cond}_{v}':q for (cond,v),q in scores.groupby(['condition','variant']) if v in ('global_only','full','ttr','ttr_raw')}
    contrasts={}
    for level in ('clean','crf23','crf35'):contrasts[level+'_Local_increment']={level+'_full':1,level+'_global_only':-1}
    for level in ('crf23','crf35'):
        contrasts[level+'_Full_drop']={level+'_full':1,'clean_full':-1}
        contrasts[level+'_Local_interaction']={level+'_full':1,level+'_global_only':-1,'clean_full':-1,'clean_global_only':1}
    groups=meta.source_group;ci=shared_contrasts(frames,pairs,groups,contrasts,iterations=1000,seed=17);atomic_csv(out/'contrasts.csv',ci)
    paper_json(out/'results.json',dict(status='completed',identity=identity,clips=len(wanted),cells=23,
        scope='fixed selected subset; original windows and clean references; codec plus pixel-format conversion; no retuned threshold',
        files={p.name:file_digest(p) for p in out.glob('*.csv*')}))


def run(root,waiting=None):
    if waiting is not None:
        wait_pid(waiting)
        state=json.loads((root/'results/runs/reference_confirmation/status.json').read_text())
        if state['status']!='reference_complete':raise RuntimeError('新参考流程没有成功完成')
    c,parent,out,e,identity=setup(root)
    def stage(probe):
        commands=[[sys.executable,'-m','evaluation.encoding_confirmation','worker','--rank',str(r),'--world','2']+(['--probe'] if probe else []) for r in (0,1)]
        ps=[subprocess.Popen(cmd,cwd=root) for cmd in commands]
        paper_json(out/'status.json',dict(status='running',phase='probe' if probe else 'full',pid=os.getpid(),children=[p.pid for p in ps]))
        try:
            while any(p.poll() is None for p in ps):
                if any(p.poll() not in (None,0) for p in ps):raise RuntimeError('编码分片失败')
                time.sleep(2)
            if any(p.returncode for p in ps):raise RuntimeError('编码分片失败')
        finally:
            for p in ps:
                if p.poll() is None:p.terminate()
            for p in ps:p.wait()
    try:
        stage(True);stage(False);evaluate(root);paper_json(out/'status.json',dict(status='completed',pid=os.getpid()))
    except BaseException as e:
        paper_json(out/'status.json',dict(status='failed',pid=os.getpid(),error=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['worker','evaluate','run']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);p.add_argument('--probe',action='store_true');p.add_argument('--wait-pid',type=int);a=p.parse_args();torch.set_num_threads(2)
    if a.action=='worker':worker(Path.cwd(),a.rank,a.world,a.probe)
    elif a.action=='run':run(Path.cwd(),a.wait_pid)
    else:evaluate(Path.cwd())
