"""只用拟合real内部留出，检查专家是否有独立统计分工。"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import paper_json
from reference import file_digest
from statistical_experts.manifests import settings
from statistical_experts.engine import load_bank
from statistical_experts.gaussian import fitted,spherical_kmeans,energy


def real_holdout(root):
    root=Path(root);c=settings(root);out=root/c['run_directory']/'real_diagnostic'
    if (out/'manifest.json').exists():return
    torch.set_num_threads(8);records=[];supports=[]
    for length in (8,16):
        bank=load_bank(root,length,'cpu');n=len(bank['ids'])
        permutation=np.random.default_rng(c['seed']).permutation(n)
        val=np.sort(permutation[:n//5]);train=np.sort(permutation[n//5:])
        centers,labels,_=spherical_kmeans(bank['context'][train],c['experts'],c['seed'])
        route=(bank['context'][val]@centers.T).argmax(dim=1)
        pool_s=fitted(bank['spatial'][train],ridge=c['ridge']);pool_t=fitted(bank['temporal'][train].flatten(0,1),ridge=c['ridge'])
        def nll(x,model):return -energy(x,model['mean'],model['chol'])+torch.log(torch.diagonal(model['chol'])).sum()
        s=bank['spatial'][val];t=bank['temporal'][val]
        baseline_s=nll(s,pool_s);baseline_t=nll(t.flatten(0,1),pool_t).reshape(len(val),length-1).mean(1)
        all_s=[];all_t=[]
        for k in range(c['experts']):
            idx=torch.as_tensor(train)[labels==k]
            fallback=len(idx)<c['min_fit_sources']
            ms=pool_s if fallback else fitted(bank['spatial'][idx],pool_s['covariance'],c['shrinkage'],c['ridge'])
            mt=pool_t if fallback else fitted(bank['temporal'][idx].flatten(0,1),pool_t['covariance'],c['shrinkage'],c['ridge'])
            all_s.append(nll(s,ms));all_t.append(nll(t.flatten(0,1),mt).reshape(len(val),length-1).mean(1))
            supports.append(dict(length=length,expert=k,fit_sources=len(idx),heldout_sources=int((route==k).sum()),fallback=fallback))
        all_s=torch.stack(all_s,1);all_t=torch.stack(all_t,1)
        for j,index in enumerate(val):
            k=int(route[j]);other=(k+1)%c['experts']
            records.append(dict(video_id=bank['ids'][index],length=length,expert=k,
                pooled_spatial_nll=float(baseline_s[j]),expert_spatial_nll=float(all_s[j,k]),other_spatial_nll=float(all_s[j,other]),
                pooled_temporal_nll=float(baseline_t[j]),expert_temporal_nll=float(all_t[j,k]),other_temporal_nll=float(all_t[j,other])))
    out.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(records);frame.to_csv(out/'heldout_scores.csv',index=False)
    frame.groupby('length')[[k for k in frame if k.endswith('_nll')]].mean().to_csv(out/'summary.csv')
    pd.DataFrame(supports).to_csv(out/'support.csv',index=False)
    paper_json(out/'manifest.json',dict(status='completed',split='同2200fit源内1760训练/440留出，8/16源划分一致',
        router_and_gaussians='全部只用1760训练源重新拟合',score='共同1024维正定Gaussian近似NLL，含logdet；GT位置先视频均值',
        interpretation='真实预测统计诊断，不是检测AUC证据；不据此重新选择已冻结超参',
        files={p.name:file_digest(p) for p in out.iterdir() if p.is_file()}))
    print(frame.groupby('length')[[k for k in frame if k.endswith('_nll')]].mean().to_string())


if __name__=='__main__':real_holdout(Path.cwd())
