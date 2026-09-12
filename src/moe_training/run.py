"""容量诊断的独立入口；已完成实验不覆盖。"""
import os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import argparse,yaml,torch


def configuration(root=ROOT):
    c=yaml.safe_load((Path(root)/'configs/moe_training.yaml').read_text())
    expected={'protocol','run_directory','source_directory','datasets','seeds','hidden_budget','experts','epochs','warmup_epochs',
              'learning_rate','weight_decay','dropout','batch_size','balance_weight','gradient_clip','diagnostic_clusters','diagnostic_seed','bootstrap_iterations','bootstrap_seed'}
    if set(c)!=expected:raise ValueError('配置存在未知或缺失字段')
    if c['hidden_budget']!=256 or c['experts']!=2 or c['epochs']!=30 or c['warmup_epochs']!=10:
        raise ValueError('本轮固定容量/训练日程改变，请另建协议')
    return c


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','analyze','verify','report'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=2);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage in ('prepare','train'):
        from moe_training import training
        getattr(training,a.stage)(ROOT,a)
    else:
        from moe_training import evaluation
        getattr(evaluation,a.stage)(ROOT,a)


if __name__=='__main__':main()
