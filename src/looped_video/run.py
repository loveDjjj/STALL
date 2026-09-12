"""示例：PYTHONPATH=src python -m looped_video.run prepare。"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import argparse
import yaml
import torch

ROOT = Path(__file__).resolve().parents[2]
FIELDS = set('protocol run_directory cache_directory source_directory folds seeds variants experiment_overrides width heads loops tile mlp_ratio epochs learning_rate weight_decay batch_size accumulation warmup_epochs amp loader_workers loader_prefetch feature_batch decode_workers prefetch_depth prefetch_gib disk_floor_gib bootstrap_replicates'.split())


def configuration(root=ROOT):
    c = yaml.safe_load((Path(root)/'configs/looped_video.yaml').read_text())
    if set(c) != FIELDS:
        raise ValueError(f'配置字段不匹配：{set(c)^FIELDS}')
    from .model import VARIANTS
    if len(set(c['variants']))!=len(c['variants']) or set(c['experiment_overrides'])-set(c['variants']):
        raise ValueError('重复实验或孤立覆盖项')
    for name in c['variants']:
        override=c['experiment_overrides'].get(name,{})
        if set(override)-{'variant','width','heads','loops'}:raise ValueError('不允许的结构覆盖')
        if override.get('variant',name) not in VARIANTS:raise ValueError('未知模型变体')
        if override.get('width',c['width'])%override.get('heads',c['heads']):raise ValueError('头维度不兼容')
        if override.get('loops',c['loops'])<1:raise ValueError('循环次数必须为正')
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare','extract','verify_cache','prepare_training_cache','train','evaluate','report'])
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='仅工程提取探针限制；0为完整范围')
    p.add_argument('--task', default='', help='指定已冻结矩阵中的单任务键')
    a = p.parse_args()
    if not 0 <= a.rank < a.world_size:
        raise ValueError('worker编号非法')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.stage == 'prepare':
        from .data import prepare
        prepare(ROOT)
    elif a.stage in ('extract','verify_cache'):
        from .cache import extract, verify_cache
        (extract if a.stage == 'extract' else verify_cache)(ROOT,a)
    elif a.stage == 'prepare_training_cache':
        from .readiness import prepare_training_cache
        prepare_training_cache(ROOT,a)
    elif a.stage == 'train':
        from .training import train
        train(ROOT,a)
    elif a.stage == 'evaluate':
        from .evaluation import evaluate
        evaluate(ROOT,a)
    else:
        from .evaluation import report
        report(ROOT,a)


if __name__ == '__main__':
    main()
