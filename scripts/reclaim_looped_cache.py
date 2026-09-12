"""回收已退役的GenVideo Uniform评价Patch；默认只生成清单。

示例：python scripts/reclaim_looped_cache.py；核对后加--apply。
唯一允许目标为下方PREFIX，不删除原视频、模型、分数或新FC缓存。
"""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from artifacts import paper_json
from reference import file_digest

PREFIX='development/genvideo/evaluation/'
BASE=ROOT/'cache/patch/benchmarks/packed_v1'
RECEIPT=ROOT/'data/catalog/looped_cache_reclamation.json'


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    index=BASE/'index.json';data=json.loads(index.read_text())
    if RECEIPT.exists():
        r=json.loads(RECEIPT.read_text())
    else:
        names=sorted(n for n in data['shards'] if n.startswith(PREFIX))
        paths=[BASE/n for n in names]
        if not paths or any(x.is_symlink() or not x.is_file() or x.resolve().parent!=(BASE/PREFIX).resolve() for x in paths):
            raise ValueError('回收目标不合法')
        r=dict(status='planned',reason='用户授权新Looped主线回收不用的旧Uniform评价缓存；新FC计划不读取该目录',
            index_sha256=file_digest(index),prefix=PREFIX,
            files=[dict(path=str(x.relative_to(ROOT)),bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns) for x in paths],
            retired_entries={k:v for k,v in data['entries'].items() if v['shard'] in names},
            recovery='不保留缓存副本；原视频和旧指标/参数不删除，可用的原视频需重新提取才能恢复Patch')
        paper_json(RECEIPT,r)
    print(r['status'],len(r['files']),sum(x['bytes'] for x in r['files'])/2**30,'GiB',flush=True)
    if not a.apply or r['status']=='removed':return
    if file_digest(index)!=r['index_sha256']:raise ValueError('旧索引改变')
    for item in r['files']:
        path=ROOT/item['path']
        if path.resolve().parent!=(BASE/PREFIX).resolve() or path.is_symlink():raise ValueError('越界')
        if path.exists():
            s=path.stat()
            if (s.st_size,s.st_mtime_ns)!=(item['bytes'],item['mtime_ns']):raise ValueError('缓存改变')
    r['status']='removing';paper_json(RECEIPT,r)
    for item in r['files']:
        (ROOT/item['path']).unlink(missing_ok=True)
    # 索引保留历史身份，以显式退役表说明不可读取的条目，不伪造旧完整缓存。
    r['status']='removed';paper_json(RECEIPT,r)
    print('回收完成；收据',RECEIPT,flush=True)


if __name__=='__main__':main()
