"""在相同帧上分离评分实现与前向batch差异，不根据指标修改官方检测器。"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from artifacts import paper_json
from reference import file_digest
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from evaluation.official_baseline import official_window,official_numpy_score


def audit(root,output,device='cuda:0'):
    root=Path(root);output=Path(output)
    cfg=yaml.safe_load((root/'configs/paper.yaml').read_text())
    source=root/'cache/contracts/official_stall/upstream_stall.py'
    spec=importlib.util.spec_from_file_location('pinned_official_stall',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    parameter=root/'precomputed/stall_params_vatex_dino_v3.npz'
    with np.load(parameter,allow_pickle=False) as z:params={k:z[k].copy() for k in z.files}
    official=mod.STALL(device,params,load_dino=False)
    extractor=AlphaStallFeatureExtractor(device,dino_repo=str(root/cfg['encoder']['repo']),
        dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    jobs=[]
    for domain in ('comgenvid','genvideo','videofeedback'):
        frame=pd.read_csv(root/'data/manifests/active'/domain/'evaluation.csv',keep_default_na=False)
        for subset in ('real','annotated'):
            r=frame[frame.subset==subset].sort_values('video_id').iloc[0]
            jobs.append((r,official_window(json.loads(r.downsample_idxs)),16))
        if domain!='comgenvid':
            frame=pd.read_csv(root/'data/manifests/short_video'/domain/'evaluation.csv',keep_default_na=False)
            for _,part in frame.groupby(['subset','source_model']):
                r=part.sort_values('video_id').iloc[0];jobs.append((r,json.loads(r.downsample_idxs),8))
    records=[]
    for r,indices,length in jobs:
        frames=decode_bounded(root/r.video_path,indices)
        prepared=extractor.prepare_frames(frames)
        extractor.pad_tail_batch=True
        g8=extractor.frames_to_global_embeddings([prepared],batch_size=8)[0]
        # 使用官方相同模型/变换的实际_embed_flat_frames，而非自行模拟其默认前向。
        official.model=extractor.model;official.transform=extractor.transform
        g32=official._embed_flat_frames(frames,batch_size=32)
        extractor.pad_tail_batch=False
        native32=extractor.frames_to_global_embeddings([prepared],batch_size=32)[0]
        if not np.array_equal(native32,g32):raise ValueError('默认batch32特征与官方前向不一致')
        a=official_numpy_score(g8[None],params);b=official._scores_from_embs(g8[None]);c=official._scores_from_embs(g32[None])
        errors=[abs(a['global_spatial_raw'][0]-b['spat_ll_agg'][0]),
                abs(a['global_temporal_raw'][0]-b['temp_ll_agg'][0]),abs(a['final_score'][0]-b['final_score'][0])]
        if not all(e==0 or np.isnan(e) for e in errors):raise ValueError('同特征官方评分不一致')
        records.append(dict(video_id=r.video_id,window_frames=length,
            feature_max_abs=float(np.max(abs(g8-g32))),same_features_score_max_abs=float(np.nanmax(errors)),
            native32_feature_error=float(np.max(abs(native32-g32))),
            batch8_score=float(b['final_score'][0]),batch32_score=float(c['final_score'][0]),
            batch_score_difference=float(c['final_score'][0]-b['final_score'][0])))
    output.mkdir(parents=True,exist_ok=False);pd.DataFrame(records).to_csv(output/'comparison.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',upstream_sha256=file_digest(source),parameters_sha256=file_digest(parameter),
        videos=len(records),scope='固定少量真实/生成视频同帧8/16帧；不是全数据batch敏感性保证',
        files={'comparison.csv':file_digest(output/'comparison.csv')}))
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':audit(Path.cwd(),Path('results/runs/complete23_official_forward_audit'))
