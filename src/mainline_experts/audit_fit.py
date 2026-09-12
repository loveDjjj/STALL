"""用少量原始fit视频核验256个D2样本的真实位置映射。"""
from src.mainline_experts.run import ROOT,configuration
import json
import numpy as np,pandas as pd,torch,yaml
from artifacts import paper_json
from reference import file_digest
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from mainline_experts.models import sample_indices


def main():
    root=ROOT;c=configuration(root);out=root/c['run_directory'];cfg=yaml.safe_load((root/'configs/paper.yaml').read_text())
    extractor=AlphaStallFeatureExtractor('cuda:0',dino_repo=str(root/cfg['encoder']['repo']),dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    results=[]
    for d in c['datasets']:
        f=pd.read_csv(root/f'results/runs/paper_fit_{d}/prepared_fit.csv',keep_default_na=False)
        # 预定首/末片段，不根据分数或模型表现选择。
        for index in (0,len(f)-1):
            row=f.iloc[index];path=root/row.feature_asset
            if file_digest(path)!=row.feature_asset_sha256:raise ValueError('fit资产损坏')
            with np.load(path) as z:g0=z['global_windows'].copy();raw0=z['raw_d2'].copy();windows=z['frame_indices'];identity=str(z['sampling_identity'])
            union=sorted(set(windows.reshape(-1).tolist()));where={x:i for i,x in enumerate(union)};pick=[[where[int(i)] for i in w] for w in windows]
            frames=extractor.prepare_frames(decode_bounded(root/row.video_path,union));features=extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0]
            g=np.stack([features['global'][p] for p in pick]);patch=torch.from_numpy(np.stack([features['patch'][p] for p in pick]))
            raw=(patch[:,2:]-2.*patch[:,1:-1]+patch[:,:-2]).reshape(-1,1024).numpy();idx=sample_indices(identity,len(windows))
            np.testing.assert_array_equal(g,g0);np.testing.assert_array_equal(raw[idx],raw0)
            results.append(dict(dataset=d,video_id=row.video_id,windows=len(windows),positions=256,global_exact=True,D2_exact=True))
    paper_json(out/'fit_sample_audit.json',dict(status='verified',cases=results,source_code=file_digest(root/'src/mainline_experts/audit_fit.py')))
    print('fit sample mapping verified',len(results),flush=True)

if __name__=='__main__':main()
