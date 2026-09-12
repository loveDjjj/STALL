"""可观测的双卡执行器：提取→验收→逐模型训练/测试→汇总。

示例：PYTHONPATH=src python -m looped_video.pipeline --adopt 123 456
仅接管明确仍在运行的本研究提取PID，不因日志沉默重新启动任务。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from artifacts import paper_json
from .run import ROOT,configuration


def command(stage,rank=0,task=''):
    result=[sys.executable,'-m','looped_video.run',stage,'--rank',str(rank),'--world-size','2']
    if task:result+=['--task',task]
    return result


def launch(out,stage,rank=0,task=''):
    log=out/'logs'/f'{stage}_{task or rank}.log';log.parent.mkdir(exist_ok=True)
    stream=log.open('a',buffering=1)
    process=subprocess.Popen(command(stage,rank,task),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
        env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    return dict(process=process,stream=stream,stage=stage,rank=rank,task=task,log=str(log))


def finish(job):
    code=job['process'].returncode;job['stream'].close()
    if code:raise RuntimeError(f"{job['stage']}退出{code}，日志{job['log']}")


def run_stages(out,stage,ranks):
    jobs=[launch(out,stage,rank) for rank in ranks]
    try:
        while jobs:
            for job in list(jobs):
                if job['process'].poll() is not None:finish(job);jobs.remove(job)
            paper_json(out/'pipeline.json',dict(status=stage,pid=os.getpid(),
                workers=[dict(pid=j['process'].pid,rank=j['rank'],stage=j['stage']) for j in jobs]))
            if jobs:time.sleep(5)
    except Exception:
        for job in jobs:
            if job['process'].poll() is None:job['process'].terminate()
        raise


def alive_extract(pid):
    p=Path(f'/proc/{pid}/cmdline')
    if not p.exists():return False
    text=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
    if not text:return False
    if 'looped_video.run extract' not in text:raise ValueError('接管PID不是本研究提取进程')
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--adopt',type=int,nargs='*',default=[]);args=p.parse_args()
    c=configuration(ROOT);out=ROOT/c['run_directory'];cache=ROOT/c['cache_directory']
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                active=[pid for pid in args.adopt if alive_extract(pid)]
                if not active:break
                paper_json(out/'pipeline.json',dict(status='extract',pid=os.getpid(),adopted_pids=active))
                time.sleep(5)
            jobs=json.loads((out/'jobs.json').read_text())['jobs']
            if any(not (cache/(j['key']+'.json')).exists() for j in jobs):
                run_stages(out,'extract',[0,1])
            if not (cache/'manifest.json').exists():run_stages(out,'prepare_training_cache',[0])
            # 用户要求不占用系统盘；所有持久资产留在/data，不建立NVMe副本。
            tasks=json.loads((out/'tasks.json').read_text())['tasks']
            tasks=sorted(tasks,key=lambda t:(c['variants'].index(t['variant']),c['folds'].index(t['fold']),c['seeds'].index(t['seed'])))
            queues={rank:[t for i,t in enumerate(tasks) if i%2==rank] for rank in (0,1)}
            active={};completed=0
            while any(queues.values()) or active:
                for rank in (0,1):
                    if rank not in active and queues[rank]:
                        task=queues[rank].pop(0);active[rank]=launch(out,'train',rank,task['key'])
                for rank,job in list(active.items()):
                    if job['process'].poll() is None:continue
                    finish(job)
                    if job['stage']=='train':active[rank]=launch(out,'evaluate',rank,job['task'])
                    else:
                        del active[rank];completed+=1
                        # 先让空闲GPU接下一模型；CPU单独汇总已完成测试，避免两个worker争写表格。
                        if queues[rank]:
                            task=queues[rank].pop(0);active[rank]=launch(out,'train',rank,task['key'])
                        if completed<len(tasks):run_stages(out,'report',[0])
                paper_json(out/'pipeline.json',dict(status='training_evaluation',pid=os.getpid(),completed=completed,total=len(tasks),
                    workers=[dict(pid=j['process'].pid,rank=r,stage=j['stage'],task=j['task']) for r,j in active.items()]))
                if active:time.sleep(5)
            run_stages(out,'report',[0])
            paper_json(out/'pipeline.json',dict(status='matrix_finished_pending_final_audit',pid=os.getpid(),completed=len(tasks)))
        except Exception as exc:
            live_jobs=locals().get('active',{})
            for job in (live_jobs.values() if isinstance(live_jobs,dict) else []):
                if job['process'].poll() is None:job['process'].terminate()
            paper_json(out/'pipeline.json',dict(status='failed',pid=os.getpid(),error=str(exc)))
            raise


if __name__=='__main__':main()
