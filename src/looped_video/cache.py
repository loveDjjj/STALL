"""FP32唯一帧缓存；双GPU分片、有界CPU预取、原子提交及内容收据。"""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import yaml
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from data.sequential_decode import decode_sequential
from data.prefetch import bounded_map, frame_reservation
from .run import configuration


def context(root):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];cache=root/c['cache_directory']
    prepared=json.loads((out/'prepared.json').read_text())
    if prepared['spec']['config']!=c:raise ValueError('配置与准备不符')
    for p,h in prepared['files'].items():
        if file_digest(out/p)!=h:raise ValueError('准备文件改变')
    cfg=yaml.safe_load((root/'configs/paper.yaml').read_text())
    spec=dict(protocol='looped_patch_float32_v1',jobs=file_digest(out/'jobs.json'),
        encoder=cfg['encoder'],batch=c['feature_batch'],pad_tail=True,dtype='float32',
        code={p:file_digest(root/p) for p in ['src/features.py','src/data/video.py','src/data/sequential_decode.py',
            'src/discriminative_moe/features.py','src/looped_video/cache.py']})
    return c,out,cache,cfg,spec,config_digest(spec)


def extract(root,args):
    root=Path(root);c,out,cache,cfg,spec,identity=context(root)
    cache.mkdir(parents=True,exist_ok=True)
    marker=cache/'identity.json'
    if marker.exists():
        prior=json.loads(marker.read_text())
        if prior!=spec:
            proof=out/'decoder_equivalence.json'
            numeric=lambda x:{k:v for k,v in x.items() if k!='code'}
            audit=json.loads(proof.read_text()) if proof.exists() else {}
            # 只允许本轮尚未训练时经过逐像素验证的解码优化；旧分片不改写身份。
            if (out/'training').exists() or numeric(prior)!=numeric(spec) or audit.get('status')!='passed':
                raise ValueError('缓存合同改变，缺少合法兼容证据')
            if audit['decoder_sha256']!=file_digest(root/'src/data/sequential_decode.py'):
                raise ValueError('解码验证与源码不符')
            for name in ('src/features.py','src/data/video.py'):
                if prior['code'][name]!=spec['code'][name]:raise ValueError('编码器或原解码被改变')
            paper_json(cache/'producer_compatibility.json',dict(previous_identity=config_digest(prior),
                current_identity=identity,previous_spec=prior,current_spec=spec,proof_sha256=file_digest(proof),
                reason='24条跨域/类别/长度逐像素一致；所有开发视频仍核验Global，失败回退原路径'))
            paper_json(marker,spec)
    else:paper_json(marker,spec)
    accepted={identity}
    if (cache/'producer_compatibility.json').exists():
        compat=json.loads((cache/'producer_compatibility.json').read_text())
        if compat['current_identity']!=identity:raise ValueError('兼容链不符')
        accepted.add(compat['previous_identity'])
    jobs=json.loads((out/'jobs.json').read_text())['jobs']
    if args.limit:
        # 探针覆盖不同域/类别/长度，不能作为正式训练的数据子集。
        selected=[];seen={}
        for j in jobs:
            tag=(j['scope'],next(iter(j.get('targets',{'external':{'subset':'external'}}).values()))['subset'],len(j['windows'][0]))
            if seen.get(tag,0)<args.limit:selected.append(j);seen[tag]=seen.get(tag,0)+1
        jobs=selected
    assigned=[j for i,j in enumerate(jobs) if i%args.world_size==args.rank]
    pending=[]
    for j in assigned:
        p=cache/(j['key']+'.json')
        if p.exists():
            r=json.loads(p.read_text());array=cache/(j['key']+'.npy')
            if r['identity'] not in accepted or file_digest(array)!=r['sha256']:raise ValueError('已有Patch损坏')
        else:pending.append(j)
    if not pending:print('Patch缓存已齐备',args.rank,flush=True);return
    if file_digest(root/cfg['encoder']['weights'])!=cfg['encoder']['weights_sha256']:raise ValueError('权重改变')
    extractor=AlphaStallFeatureExtractor(f'cuda:{args.rank%2}',dino_repo=str(root/cfg['encoder']['repo']),
        dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    def load(j):
        path=root/j['video_path'];s=path.stat()
        if (s.st_size,s.st_mtime_ns)!=(j['source_bytes'],j['source_mtime_ns']):raise ValueError('原视频改变')
        indices=sorted({i for w in j['windows'] for i in w})
        return extractor.prepare_frames(decode_sequential(path,indices))
    stream=bounded_map(pending,load,lambda j:frame_reservation(root/j['video_path'],len({i for w in j['windows'] for i in w})),
        workers=c['decode_workers'],depth=c['prefetch_depth'],budget=c['prefetch_gib']*2**30)
    start=time.perf_counter();frames_done=0;fallback_count=0
    from discriminative_moe.features import projection,global_descriptors
    basis=projection()
    def matches(f,j,pick):
        g=np.stack([f['global'][p] for p in pick])
        if j['scope']=='development':
            with np.load(root/'results/runs/mainline_experts/global'/(j['key']+'.npz')) as z:
                return np.array_equal(g,z['global_features'])
        a,t=global_descriptors(g,basis)
        with np.load(root/c['source_directory']/'external/features'/(j['key']+'.npz')) as z:
            return np.array_equal(a,z['G']) and np.array_equal(t,z['T'])
    try:
        for i,(j,frames) in enumerate(stream,1):
            if shutil.disk_usage(cache).free < c['disk_floor_gib']*2**30+frames.shape[0]*196*1024*4:
                raise RuntimeError('磁盘达到预留边界；已完成缓存保留，可清理后续跑')
            f=extractor.frames_to_global_patch_embeddings([frames],batch_size=c['feature_batch'])[0]
            indices=sorted({x for w in j['windows'] for x in w});where={x:i for i,x in enumerate(indices)}
            pick=[[where[x] for x in w] for w in j['windows']]
            if not matches(f,j,pick):
                strict=extractor.prepare_frames(decode_bounded(root/j['video_path'],indices))
                f=extractor.frames_to_global_patch_embeddings([strict],batch_size=c['feature_batch'])[0]
                fallback_count+=1
                if not matches(f,j,pick):raise ValueError('原路径亦不能复现Global：'+j['key'])
            x=np.asarray(f['patch'])
            if x.dtype!=np.float32 or x.shape!=(len(indices),196,1024) or not np.isfinite(x).all():
                raise ValueError('Patch格式/数值不正确')
            p=cache/(j['key']+'.npy');temp=cache/(j['key']+'.tmp.npy')
            np.save(temp,x,allow_pickle=False);temp.replace(p)
            s=p.stat()
            paper_json(cache/(j['key']+'.json'),dict(identity=identity,key=j['key'],scope=j['scope'],
                sha256=file_digest(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,frame_indices=indices,
                window_positions=pick,shape=list(x.shape),global_exact=j['scope']=='development',
                external_summary_exact=j['scope']=='external'))
            frames_done+=len(indices)
            if i%32==0 or i==len(pending):
                elapsed=time.perf_counter()-start
                paper_json(cache/f'progress_{args.rank}.json',dict(completed=i,total=len(pending),seconds=elapsed,
                    frames=frames_done,videos_per_second=i/elapsed,eta_seconds=(len(pending)-i)*elapsed/i,
                    strict_fallbacks=fallback_count))
                print('Patch',args.rank,i,len(pending),round(i/elapsed,2),'条/秒',flush=True)
    finally:stream.close()
    paper_json(cache/f'worker_{args.rank}.json',dict(status='extracted',identity=identity,
        completed=len(pending),assigned=len(assigned),seconds=time.perf_counter()-start,probe_limit=args.limit))


def verify_cache(root,args):
    c,out,cache,cfg,spec,identity=context(root)
    jobs=json.loads((out/'jobs.json').read_text())['jobs'];records={}
    accepted={identity}
    if (cache/'producer_compatibility.json').exists():
        compat=json.loads((cache/'producer_compatibility.json').read_text())
        if compat['current_identity']!=identity:raise ValueError('兼容链不符')
        accepted.add(compat['previous_identity'])
    for i,j in enumerate(jobs,1):
        r=json.loads((cache/(j['key']+'.json')).read_text());p=cache/(j['key']+'.npy')
        if r['identity'] not in accepted or file_digest(p)!=r['sha256']:raise ValueError('缓存内容不符')
        x=np.load(p,mmap_mode='r',allow_pickle=False)
        indices=sorted({i for w in j['windows'] for i in w})
        expected=[[indices.index(i) for i in w] for w in j['windows']]
        if r['frame_indices']!=indices or r['window_positions']!=expected:raise ValueError('帧索引改变')
        if list(x.shape)!=r['shape'] or x.dtype!=np.float32:raise ValueError('数组格式不符')
        if (p.stat().st_size,p.stat().st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('缓存stat改变')
        records[j['key']]=r
        if i%2000==0:print('verify',i,len(jobs),flush=True)
    paper_json(cache/'manifest.json',dict(status='verified',identity=identity,records=records,jobs=len(jobs)))
    print('完整Patch内容hash与身份核验通过',len(jobs),flush=True)
