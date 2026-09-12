"""独立监督实验入口；不触及原paper配置与完成资产。"""
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

import argparse
import yaml
import torch


def configuration(root=ROOT):
    c=yaml.safe_load((Path(root)/'configs/discriminative_moe.yaml').read_text())
    if c['expert_candidates'] != [2,4] or c['projection_dimension'] != 64:
        raise ValueError('本轮容量/表示合同已改变，需新协议')
    return c


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['prepare','local','train','evaluate','analyze','verify','report','external_features','external_evaluate'])
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=2)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.stage=='prepare':
        from discriminative_moe.data import prepare
        prepare(ROOT)
    elif a.stage=='local':
        from discriminative_moe.extract import extract
        extract(ROOT,a)
    elif a.stage=='train':
        from discriminative_moe.training import train
        train(ROOT,a)
    elif a.stage=='evaluate':
        from discriminative_moe.evaluation import evaluate
        evaluate(ROOT,a)
    elif a.stage=='analyze':
        from discriminative_moe.analysis import analyze
        analyze(ROOT,a)
    elif a.stage=='verify':
        from discriminative_moe.audit import verify
        verify(ROOT,a)
    elif a.stage=='report':
        from discriminative_moe.reporting import report
        report(ROOT,a)
    elif a.stage=='external_features':
        from discriminative_moe.external import extract
        extract(ROOT,a)
    elif a.stage=='external_evaluate':
        from discriminative_moe.external import evaluate
        evaluate(ROOT,a)
    else:
        raise NotImplementedError('当前阶段尚在实现：'+a.stage)


if __name__=='__main__':main()
