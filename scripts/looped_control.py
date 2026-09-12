"""可重复的后台调度命令；只控制本研究精确PID，不广泛清理Python进程。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from artifacts import paper_json
from looped_video.run import configuration


def load(path):return json.loads(path.read_text()) if path.exists() else {}


def process_info(pid):
    if not pid:return None
    p=Path(f'/proc/{pid}')
    try:
        cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
        cwd=p.joinpath('cwd').resolve()
        if not cmd or cwd!=ROOT:return None
        if 'looped_video.group_train' not in cmd and 'multiprocessing.spawn' not in cmd:return None
        return dict(pid=pid,cmd=cmd)
    except (FileNotFoundError,PermissionError,ProcessLookupError):return None


def current(out):
    state=load(out/'pipeline.json');launched=load(out/'launcher.json')
    ids={state.get('pid'),launched.get('pid')}|{w.get('pid') for w in state.get('workers',[])}
    return state,[p for pid in ids if (p:=process_info(pid))]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','status','stop','report','plan'],default='status',nargs='?')
    args=parser.parse_args();c=configuration(ROOT);out=ROOT/c['run_directory'];out.mkdir(parents=True,exist_ok=True)
    state,live=current(out)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'NUMPY_MADVISE_HUGEPAGE':'0','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'2'}
    if args.action=='start':
        if live:
            print('已有训练/worker，不重复启动：',json.dumps(live,ensure_ascii=False));return
        if not (ROOT/c['cache_directory']/'manifest.json').exists():raise RuntimeError('缺少Patch训练准入记录')
        command=[sys.executable,'-m','looped_video.group_train']
        with (out/'group_training.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        paper_json(out/'launcher.json',dict(pid=proc.pid,command=command,start_unix=time.time(),
            log='group_training.log',detached=True,environment={k:env[k] for k in ['NUMPY_MADVISE_HUGEPAGE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
        time.sleep(1)
        if proc.poll() is not None:raise RuntimeError('调度器提前结束，请查group_training.log')
        print('后台自动矩阵已启动，PID',proc.pid)
    elif args.action=='stop':
        parents=[p for p in live if 'looped_video.group_train' in p['cmd']]
        if parents:
            for p in parents:os.kill(p['pid'],signal.SIGTERM)
            print('已请求父进程停止；保留各模型最近25更新检查点。')
        elif live:
            for p in live:os.kill(p['pid'],signal.SIGTERM)
            print('已停止身份明确的遗留worker。')
        else:print('没有活跃训练。')
    elif args.action=='report':
        if live:raise RuntimeError('运行中由调度器自动汇总，请用status查看；避免争写结果')
        subprocess.run([sys.executable,'-m','looped_video.run','report'],cwd=ROOT,env=env,check=True)
    elif args.action=='plan':
        tasks=load(out/'tasks.json')['tasks'];print('模型',len(tasks),'组',len(c['seeds'])*len(c['folds']))
        for seed in c['seeds']:
            for fold in c['folds']:print(seed,fold,' → 20epoch → 最佳验证checkpoint → 测试/指标/区间/验收 → 下一组')
    else:
        print(json.dumps(dict(pipeline=state,live_processes=live,
            models_trained=len(list((out/'training').glob('*/manifest.json'))),
            models_evaluated=len(list((out/'evaluation').glob('*/manifest.json'))),
            groups_passed=len(list((out/'groups').glob('*/postprocess.json')))),ensure_ascii=False,indent=2))
        group=state.get('group')
        if group:
            for name in ['progress.json','io_progress.json']:
                p=out/'groups'/group/name
                if p.exists():print(name,json.dumps(load(p),ensure_ascii=False))


if __name__=='__main__':main()
