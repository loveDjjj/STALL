"""十模型实际输入尺寸的显存/计算探针，不使用生产标签或改变正式权重。"""
import time
from pathlib import Path
import torch
from artifacts import paper_json
from looped_video.run import configuration,ROOT
from looped_video.training import model_arguments
from looped_video.model import LoopedDetector

torch.set_num_threads(2);torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False
c=configuration();models=[]
for name in c['variants']:
    torch.manual_seed(17)
    model=LoopedDetector(**model_arguments(c,name)).cuda()
    models.append((name,model,torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])))
x=torch.randn(24,16,196,1024,device='cuda');valid=torch.ones(24,16,dtype=torch.bool,device='cuda')
owners=torch.arange(8,device='cuda').repeat_interleave(3);y=torch.arange(8,device='cuda').remainder(2).float()
rows=[]
for repeat in range(2):
    for name,model,opt in models:
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x,valid,owners,8);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();rows.append(dict(variant=name,repeat=repeat,seconds=time.perf_counter()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,loss=float(loss.detach())))
        print(name,repeat,round(rows[-1]['seconds'],3),round(rows[-1]['peak_allocated_gib'],2),flush=True)
paper_json(ROOT/c['run_directory']/'matrix_gpu_probe.json',dict(status='passed',input=[24,16,196,1024],
    scope='单卡合成输入，十模型同时驻留但依次计算；不包含磁盘、DDP通信或生产训练',rows=rows))
