"""写入可复现实验产物，默认禁止覆盖已有结果。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import config_digest, dump_config


def paper_json(path: Path, value: dict) -> None:
    """新产物原子写；Global正无穷用明确字符串保存，不输出非标准JSON数值。"""
    import math
    import os
    import tempfile
    def convert(item):
        if isinstance(item, dict):return {str(k):convert(v) for k,v in item.items()}
        if isinstance(item, (list,tuple)):return [convert(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            if item==math.inf:return 'Infinity'
            raise ValueError('产物中出现NaN或负无穷')
        return item
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name,suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(convert(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def checkpoint_write(path, identity, payload):
    """先按真实JSON表示归一化，再保存载荷hash，避免Infinity表示改变校验值。"""
    import math
    def safe(value):
        if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):
            if value==math.inf:return 'Infinity'
            raise ValueError('checkpoint包含非法非有限值')
        return value
    payload=safe(payload)
    paper_json(Path(path),dict(schema='paper_checkpoint_v1',identity=identity,payload=payload,payload_sha256=config_digest(payload)))


def checkpoint_read(path, identity, video_id):
    value=json.loads(Path(path).read_text())
    if value.get('schema')!='paper_checkpoint_v1' or value.get('identity')!=identity:
        raise ValueError('checkpoint协议身份不匹配')
    payload=value['payload']
    if config_digest(payload)!=value['payload_sha256'] or payload.get('video_id')!=video_id:
        raise ValueError('checkpoint内容损坏或视频身份错误')
    return payload


def atomic_csv(path, frame):
    """阶段暂存表先完整写入再替换；调用方负责确保该暂存路径属于当前run。"""
    import os
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name,suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            frame.to_csv(stream,index=False);stream.flush();os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    temporary.replace(path)


def publish_stage_file(source,destination,expected_hash):
    """独占发布阶段产物；中断恢复仅接受已有的同内容文件，不覆盖用户修改。"""
    import os
    from reference import file_digest
    source=Path(source);destination=Path(destination)
    if file_digest(source)!=expected_hash:raise ValueError('阶段暂存产物hash改变')
    if destination.is_symlink():raise ValueError('拒绝发布到既有符号链接')
    if destination.exists():
        if file_digest(destination)!=expected_hash:raise ValueError('已发布产物与阶段内容不一致，拒绝覆盖')
        return
    os.link(source,destination)


def resume_paper_run(root,config,inputs,directory,stage):
    """拒绝恢复时改代码/配置/输入；不放宽为同名文件或近似参数。"""
    from reference import file_digest
    directory=Path(directory).resolve();root=Path(root)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    expected_inputs=manifest.get('stage_inputs',{}).get(stage,manifest['inputs'])
    if manifest['config_sha256']!=config_digest(config) or expected_inputs!=inputs:
        raise ValueError('恢复配置或输入身份改变')
    if manifest.get('action')!=stage:raise ValueError('恢复阶段不匹配或旧run没有恢复合同')
    for name,digest in manifest['code_sha256'].items():
        if file_digest(root/name)!=digest or file_digest(directory/'code_snapshot'/name)!=digest:
            raise ValueError('恢复源码改变，请使用原源码快照或新建run')
    complete=stage in manifest.get('completed_steps',[]) and (directory/'artifact_manifest.json').is_file()
    if complete:
        state=json.loads((directory/'artifact_manifest.json').read_text())
        if state.get('status')!='completed':raise ValueError('完成标记与产物状态不一致')
        for name,digest in state['artifacts'].items():
            if file_digest(directory/name)!=digest:raise ValueError('已完成run的产物改变')
    return directory,complete


def create_paper_run(root, config, command, inputs, output=None):
    """一个输出目录只创建一次；后续恢复需要单独的身份验证，不能当作覆盖开关。"""
    import shlex
    from reference import file_digest
    root=Path(root)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    digest=config_digest(config)
    directory=Path(output) if output else root/config['output']['root']/f'{config["protocol"]["id"]}__{stamp}__{digest[:8]}'
    directory=directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    dump_config(directory/'resolved_config.yaml',config)
    (directory/'command.txt').write_text(shlex.join(command)+'\n')
    files=['src/config.py','src/artifacts.py','src/reference.py','src/reference_fit.py','src/execution.py','src/selection.py','src/workflow.py',
           'src/features.py','src/math_utils.py','src/data/video.py','src/data/prefetch.py','src/branches/global_branch.py',
           'src/evaluation/metrics.py','src/evaluation/tables.py','src/evaluation/bootstrap.py','src/data/manifest.py',
           'src/data/__init__.py','src/branches/__init__.py','src/evaluation/__init__.py','scripts/run.py','configs/paper.yaml']
    code_hashes={name:file_digest(root/name) for name in files}
    for name,source_digest in code_hashes.items():
        target=directory/'code_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
        if file_digest(target)!=source_digest or file_digest(root/name)!=source_digest:raise ValueError('保存run源码期间文件改变')
    # 上游run可能继续追加评价阶段，保留消费时的manifest版本，避免只剩无法还原的旧hash。
    input_snapshots={}
    def freeze_inputs(value):
        if not isinstance(value,dict):return
        source_path=expected=None
        if value.get('manifest_sha256') and value.get('path'):
            source_path=Path(value['path'])/'run_manifest.json';expected=value['manifest_sha256']
        elif value.get('source_manifest_sha256') and value.get('source_run'):
            source_path=Path(value['source_run'])/'run_manifest.json';expected=value['source_manifest_sha256']
        if source_path is not None:
            if file_digest(source_path)!=expected:raise ValueError('输入run的manifest在消费期间改变')
            target=directory/'input_manifests'/f'{expected}.json'
            target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copy2(source_path,target)
            if file_digest(target)!=expected:raise ValueError('输入manifest副本hash不符')
            input_snapshots[expected]=dict(source=str(source_path),path=str(target.relative_to(directory)))
        for item in value.values():
            if isinstance(item,dict):freeze_inputs(item)
    freeze_inputs(inputs)
    patch=subprocess.check_output(['git','-C',str(root),'diff','--binary','HEAD'])
    (directory/'code_snapshot/git_tracked.patch').write_bytes(patch)
    git_state=subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1'])
    (directory/'code_snapshot/git_status.txt').write_bytes(git_state)
    import importlib.metadata
    versions={}
    for package in ('numpy','pandas','torch','torchvision','PyYAML','scikit-learn'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not-installed'
    manifest=dict(protocol_id=config['protocol']['id'],config_sha256=digest,command=command,inputs=inputs,
                  code_sha256=code_hashes,code_snapshot='code_snapshot',git_commit=git_commit(root),
                  git_dirty=bool(git_state.strip()),git_diff_sha256=file_digest(directory/'code_snapshot/git_tracked.patch'),
                  environment=versions,created_utc=stamp,
                  input_manifest_snapshots=input_snapshots,
                  completed_steps=[],scope='paper-v1; stages are recorded separately')
    paper_json(directory/'run_manifest.json',manifest)
    return directory


class PaperLog:
    """单进程CLI同时保留终端输出与日志；退出时恢复调用方流。"""
    def __init__(self,directory):self.path=Path(directory)/'logs/console.log'
    def __enter__(self):
        import sys
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.path.open('a');self.stdout=sys.stdout;self.stderr=sys.stderr
        class Tee:
            def __init__(self,original,log):self.original=original;self.log=log
            def write(self,text):self.log.write(text);self.log.flush();return self.original.write(text)
            def flush(self):self.log.flush();self.original.flush()
            def isatty(self):return False
        sys.stdout=Tee(self.stdout,self.log);sys.stderr=Tee(self.stderr,self.log)
        return self
    def __exit__(self,*args):
        import sys
        sys.stdout=self.stdout;sys.stderr=self.stderr;self.log.close()


def finish_paper_stage(directory, stage):
    from reference import file_digest
    directory=Path(directory)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    if stage not in manifest['completed_steps']:manifest['completed_steps'].append(stage)
    paper_json(directory/'run_manifest.json',manifest)
    artifacts={str(p.relative_to(directory)):file_digest(p) for p in directory.rglob('*')
               if p.is_file() and p.name not in ('artifact_manifest.json','status.json','.run.lock') and p.suffix not in ('.log','.tmp')}
    paper_json(directory/'artifact_manifest.json',dict(status='completed',stage=stage,artifacts=artifacts))
    paper_json(directory/'status.json',dict(status='completed',stage=stage,completed_steps=manifest['completed_steps']))


def read_paper_scores(directory,variant=None):
    """读取已完成评分或组件表的显式变体，禁止把多变体混成重复视频集合。"""
    from reference import file_digest
    directory=Path(directory)
    manifest=json.loads((directory/'run_manifest.json').read_text())
    steps=set(manifest.get('completed_steps',[]))
    components='components' in steps
    if not steps & {'score','replay','components','import_reference'}:raise ValueError('该run没有完成评分')
    if 'import_reference' in steps and manifest.get('schema')!='paper_score_reference_v1':raise ValueError('分数参考schema错误')
    if components and variant is None:raise ValueError('组件run必须指定variant')
    if not components and variant is not None:raise ValueError('普通评分run不接受variant')
    if manifest.get('code_snapshot'):
        for name,digest in manifest['code_sha256'].items():
            if file_digest(directory/manifest['code_snapshot']/name)!=digest:raise ValueError('run源码快照改变')
    for digest,snapshot in manifest.get('input_manifest_snapshots',{}).items():
        if file_digest(directory/snapshot['path'])!=digest:raise ValueError('输入manifest快照改变')
    state=json.loads((directory/'artifact_manifest.json').read_text())
    name='video_scores.csv.gz' if components else 'video_scores.csv'
    path=directory/name
    if state.get('status')!='completed' or state['artifacts'].get(name)!=file_digest(path):raise ValueError('评分产物不完整或改变')
    frame=pd.read_csv(path,float_precision='round_trip',dtype={'video_id':str})
    if components:
        if variant not in set(frame.variant):raise ValueError('组件variant不存在')
        frame=frame[frame.variant==variant].copy()
    return frame,dict(path=str(directory),variant=variant,manifest_sha256=file_digest(directory/'run_manifest.json'),scores_sha256=file_digest(path))


def git_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
