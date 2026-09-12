"""复用写入后完整SHA256收据，完整核验结构/身份；不冒充二次全盘复核。"""
import json
import re
import time
from pathlib import Path
import numpy as np
from artifacts import paper_json
from reference import file_digest
from .cache import context


def validate_entry(cache,j,r,accepted):
    p=cache/(j['key']+'.npy')
    if r['identity'] not in accepted or r['key']!=j['key'] or r['scope']!=j['scope']:
        raise ValueError('缓存生产者或视频身份不符')
    if not re.fullmatch(r'[0-9a-f]{64}',r.get('sha256','')):
        raise ValueError('缺少写入后完整内容hash')
    s=p.stat()
    if (s.st_size,s.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):
        raise ValueError('缓存自写入后发生变化；必须单独重新检查：'+j['key'])
    indices=sorted({i for w in j['windows'] for i in w})
    positions={v:i for i,v in enumerate(indices)}
    expected=[[positions[i] for i in w] for w in j['windows']]
    if r['frame_indices']!=indices or r['window_positions']!=expected:
        raise ValueError('窗口/帧索引不符')
    x=np.load(p,mmap_mode='r',allow_pickle=False)
    if x.dtype!=np.float32 or x.shape!=(len(indices),196,1024) or list(x.shape)!=r['shape']:
        raise ValueError('Patch头部格式不符')
    # 每个文件的正文有限值在生产者写入前检查；这里不重复扫描正文。
    if j['scope']=='development' and not r.get('global_exact'):
        raise ValueError('开发Global未通过回归')


def prepare_training_cache(root,args):
    root=Path(root);c,out,cache,cfg,spec,identity=context(root)
    if (out/'training').exists():raise ValueError('训练开始后不能改写缓存清单')
    jobs=json.loads((out/'jobs.json').read_text())['jobs'];accepted={identity}
    compatibility=cache/'producer_compatibility.json'
    if compatibility.exists():
        proof=json.loads(compatibility.read_text())
        if proof['current_identity']!=identity:raise ValueError('生产者兼容链错误')
        if proof['proof_sha256']!=file_digest(out/'decoder_equivalence.json'):raise ValueError('解码兼容证据改变')
        accepted.add(proof['previous_identity'])
    log=out/'logs/verify_cache_0.log'
    matches=re.findall(r'^verify (\d+) (\d+)$',log.read_text(),re.M) if log.exists() else []
    prefix=max((int(a) for a,b in matches if int(b)==len(jobs)),default=0)
    records={};start=time.perf_counter()
    for i,j in enumerate(jobs,1):
        r=json.loads((cache/(j['key']+'.json')).read_text())
        validate_entry(cache,j,r,accepted)
        source=(root/j['video_path']).stat()
        if (source.st_size,source.st_mtime_ns)!=(j['source_bytes'],j['source_mtime_ns']):
            raise ValueError('原始视频身份改变')
        records[j['key']]=r
        if i%1000==0:
            paper_json(out/'cache_readiness_progress.json',dict(completed=i,total=len(jobs),seconds=time.perf_counter()-start))
            print('身份/头部检查',i,len(jobs),flush=True)
    receipt=dict(status='ready_for_training',identity=identity,records=records,jobs=len(jobs),
        verification=dict(mode='producer_full_sha256_readback_plus_all_stat_headers_indices',
            secondary_full_rehash_complete=False,secondary_logged_verified_prefix=prefix,
            secondary_log_sha256=file_digest(log) if log.exists() else None,
            decision='用户明确要求足够时进入训练；复用生成时逐文件完整hash，不将二次复核描述为全部完成',
            code_sha256=file_digest(Path(__file__)),seconds=time.perf_counter()-start))
    paper_json(cache/'manifest.json',receipt)
    paper_json(out/'verification_transition.json',receipt['verification'])
    print('训练准入通过',len(jobs),'；二次整库重读未完成，不作为训练阻塞条件',flush=True)
