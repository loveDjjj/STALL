"""仅迁移未完成训练的传输实现身份；逐张量核验模型/优化器/进度不变。"""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from .run import ROOT,configuration
from .ddp_train import task_spec
from .training import save_torch


def fingerprint(state):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,torch.Tensor):
            h.update(str((str(x.dtype),tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda y:(str(type(y)),str(y))):add(k);add(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(str(type(x)).encode());h.update(str(len(x)).encode())
            for v in x:add(v)
        else:h.update(repr((type(x).__name__,x)).encode())
    add({k:v for k,v in state.items() if k!='identity'})
    return h.hexdigest()


def main():
    c=configuration(ROOT);out=ROOT/c['run_directory']
    test=out/'buffer_tests.log'
    if not test.exists() or '27 passed' not in test.read_text():raise ValueError('缺少本次固定缓冲完整测试')
    meta=pd.read_csv(out/'videos.csv',keep_default_na=False);roles=pd.read_csv(out/'roles.csv',keep_default_na=False)
    for dest in (out/'training').iterdir():
        if not (dest/'identity.json').exists():continue
        if (dest/'manifest.json').exists():raise ValueError('禁止改写已完成模型的身份')
        old=json.loads((dest/'identity.json').read_text());task=old['task']
        s=roles[roles.fold.eq(task['fold'])].set_index('video_id').loc[meta.video_id]
        tm=meta.loc[s.role.to_numpy()=='train'].reset_index(drop=True);vm=meta.loc[s.role.to_numpy()=='validation'].reset_index(drop=True)
        new=task_spec(ROOT,out,c,task,tm,vm,old['runtime'])
        if old==new:continue
        strip=lambda x:{k:v for k,v in x.items() if k not in ('code','transport')}
        if strip(old)!=strip(new):raise ValueError('修改超出传输层：数据/模型配置/抽样不能改变')
        for name,h in old['code'].items():
            if name!='src/looped_video/ddp_train.py' and new['code'].get(name)!=h:
                raise ValueError('非授权源码改变：'+name)
        oldid=config_digest(old);newid=config_digest(new)
        receipt=dict(status='migrating',previous_spec=old,current_spec=new,test_sha256=file_digest(test),checkpoints={})
        journal=dest/'transport_migration.json';paper_json(journal,receipt)
        for name in ['last.pt','model.pt']:
            p=dest/name
            if not p.exists():continue
            backup=dest/(name+'.pre_transport')
            if not backup.exists():shutil.copyfile(p,backup)
            state=torch.load(backup,map_location='cpu',weights_only=True)
            if state['identity']!=oldid:raise ValueError('原检查点身份错误')
            before=fingerprint(state);state['identity']=newid;save_torch(p,state)
            restored=torch.load(p,map_location='cpu',weights_only=True)
            if restored['identity']!=newid or fingerprint(restored)!=before:raise ValueError('检查点数值变化')
            receipt['checkpoints'][name]=dict(previous_file_sha256=file_digest(backup),current_file_sha256=file_digest(p),
                numerical_state_sha256=before,epoch=state.get('epoch'),next_update=state.get('next_update'))
        paper_json(dest/'identity.json',new);receipt['status']='completed';paper_json(journal,receipt)
        print('模型/优化器/进度逐张量一致',task['key'],receipt['checkpoints'],flush=True)


if __name__=='__main__':main()
