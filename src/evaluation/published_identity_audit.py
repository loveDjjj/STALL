"""与官方公布身份名单作集合比较，不把数量相近当成同数据。"""
from pathlib import Path
import pandas as pd
from artifacts import paper_json
from reference import file_digest


def audit(root,upstream):
    root=Path(root);upstream=Path(upstream)
    ev=root/'data/manifests/active/comgenvid/evaluation.csv';fit=root/'data/manifests/active/comgenvid/fit.csv'
    frame=pd.read_csv(ev);fitted=pd.read_csv(fit)
    results=[];sources={str(ev):file_digest(ev),str(fit):file_digest(fit)}
    for model,name in [('MSVD','msvd_sampled_videos.csv'),('Sora','sora_sampled_videos.csv'),('VEO3','veo3_sampled_videos.csv')]:
        p=upstream/'supplementary/csvs'/name;sources[str(p)]=file_digest(p)
        published=set(pd.read_csv(p).filename)
        current=set(frame.loc[frame.source_model.eq(model),'video_path'].map(lambda s:Path(s).name))
        fit_names=set(fitted.video_path.map(lambda s:Path(s).name)) if model=='MSVD' else set()
        results.append(dict(dataset='comgenvid',source=model,published_unique=len(published),current_pool=len(current),
            intersection=len(published&current),current_outside_published=len(current-published),
            published_fit_overlap=len(published&fit_names),published_not_eval_or_fit=len(published-current-fit_names)))
    out=root/'results/runs/complete23_published_identity_audit';out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
    paper_json(out/'manifest.json',dict(status='completed',inputs=sources,
        scope='文件名身份交集；不是视频内容哈希或官方实际平衡抽样身份验证',
        files={'comparison.csv':file_digest(out/'comparison.csv')}))


if __name__=='__main__':audit(Path.cwd(),Path('/tmp/alpha_stall_related_STALL'))
