"""真实参考拟合角色隔离、身份、Gaussian/CDF阶段恢复与导出合同。"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import load_paper_config
from math_utils import StableGaussianParams
from reference import file_digest,load_bundle
from reference_fit import prepare_fit_inputs,fit_from_assets,verified_fit_bundle
from selection import uniform_windows
import reference_fit
import workflow


@pytest.fixture
def fit_case(tmp_path):
    config=load_paper_config(ROOT/'configs/paper.yaml')
    weights=tmp_path/'weights';weights.write_bytes(b'fixture')
    config['encoder']['weights']=str(weights);config['encoder']['weights_sha256']=file_digest(weights)
    config['runtime']['device']='cpu'
    config['runtime']['prefetch_enabled']=False
    config_path=tmp_path/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    paths={}
    for role in ('fit','cdf','evaluation','threshold'):
        records=[]
        for i in range(2):
            vid=f'{role}_{i}';video=tmp_path/f'{vid}.mp4';video.write_bytes(vid.encode())
            row=dict(video_id=vid,video_path=str(video),source_group=vid,dataset='test' if role in ('fit','evaluation') else 'vatex',
                     subset='real',split=role,downsample_idxs=json.dumps(list(range(32))))
            if role=='fit':
                asset=tmp_path/f'{vid}.npz';rng=np.random.default_rng(i)
                g=rng.normal(size=(3,16,1024)).astype(np.float32)
                g[:,1]=g[:,0]  # Global严格零T1不进入拟合。
                raw=rng.normal(size=(256,1024)).astype(np.float32);raw[0]=0  # Local零向量参与。
                np.savez_compressed(asset,video_id=vid,global_windows=g,raw_d2=raw,
                    frame_indices=np.asarray([w.frame_indices for w in uniform_windows(range(32),3)]))
                row.update(feature_asset=str(asset),feature_asset_sha256=file_digest(asset))
            records.append(row)
        paths[role]=tmp_path/f'{role}.csv';pd.DataFrame(records).to_csv(paths[role],index=False)
    return config,config_path,paths


def test_roles_and_cached_vector_identity(fit_case,monkeypatch):
    config,_,paths=fit_case
    fit,cdf,inputs=prepare_fit_inputs(ROOT,config,'test',*paths.values())
    observed=[]
    def record(videos,**kwargs):
        observed.append(videos)
        return StableGaussianParams(np.zeros(1024),np.eye(1024),np.empty(0))
    monkeypatch.setattr(reference_fit,'fit_gaussian',record)
    params,stats=fit_from_assets(ROOT,fit)
    assert set(params)=={'gs','gt','lt'}
    assert stats['gs']['observations']==96 and stats['gt']['observations']==84 and stats['lt']['observations']==512
    assert np.count_nonzero(observed[2][0][0])==0
    np.testing.assert_allclose(np.linalg.norm(observed[2][0][1:],axis=1),1,atol=2e-7)
    broken=pd.read_csv(paths['cdf']);broken.loc[0,'source_group']='fit_0';broken.to_csv(paths['cdf'],index=False)
    with pytest.raises(ValueError,match='相同source_group'):prepare_fit_inputs(ROOT,config,'test',*paths.values())


def test_fit_rejects_fake_and_changed_asset(fit_case):
    config,_,paths=fit_case
    original=pd.read_csv(paths['fit']);bad=original.copy();bad.loc[0,'subset']='annotated';bad.to_csv(paths['fit'],index=False)
    with pytest.raises(ValueError,match='真实'):prepare_fit_inputs(ROOT,config,'test',*paths.values())
    original.loc[0,'feature_asset_sha256']='0'*64;original.to_csv(paths['fit'],index=False)
    with pytest.raises(ValueError,match='hash'):prepare_fit_inputs(ROOT,config,'test',*paths.values())


def test_fit_cli_pause_resume_and_export(fit_case,tmp_path,monkeypatch):
    config,config_path,paths=fit_case
    spec=importlib.util.spec_from_file_location('fit_cli',ROOT/'scripts/run.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    run=tmp_path/'run'
    args=['run.py','--config',str(config_path),'fit','--dataset','test','--output',str(run)]
    for role,path in paths.items():args += [f'--{role}-manifest',str(path)]
    fits=[];scored=[];fail=[True]
    def fit_model(videos,**kwargs):
        fits.append(len(videos));return StableGaussianParams(np.zeros(1024),np.eye(1024),np.empty(0))
    monkeypatch.setattr(reference_fit,'fit_gaussian',fit_model)
    class Scorer:
        def __init__(self,*args,**kwargs):pass
        def score_global_first_window(self,path,indices):
            return dict(windows=[dict(global_spatial_raw=0.,global_temporal_raw=float('inf'))])
        def score_raw(self,path,indices,**kwargs):
            scored.append(Path(path).stem)
            if Path(path).stem=='cdf_1' and fail[0]:raise RuntimeError('模拟CDF中断')
            return dict(windows=[dict(local_raw=float(i)) for i in range(3)])
    monkeypatch.setattr(workflow,'RawVideoScorer',Scorer)
    monkeypatch.setattr(sys,'argv',args+['--dry-run']);cli.main()
    assert not run.exists() and not fits
    monkeypatch.setattr(sys,'argv',args+['--stop-after','gaussian']);cli.main()
    assert json.loads((run/'status.json').read_text())['status']=='paused'
    assert fits==[2,2,2] and not (run/'references').exists()
    with pytest.raises(ValueError,match='未完成CDF'):verified_fit_bundle(run)
    monkeypatch.setattr(sys,'argv',args+['--resume'])
    with pytest.raises(RuntimeError,match='模拟CDF中断'):cli.main()
    assert (run/'cdf_raw/000001.json').is_file() and not (run/'references').exists()
    fail[0]=False;scored.clear()
    cli.main()
    assert scored==['cdf_1'] and fits==[2,2,2]
    bundle=load_bundle(run/'references/test.npz')
    assert np.isposinf(bundle.global_temporal.calibration_raw).all()
    np.testing.assert_array_equal(bundle.local_cdfs[3],[1.,1.])
    before=file_digest(run/'references/test.npz');cli.main()
    assert file_digest(run/'references/test.npz')==before and scored==['cdf_1']
    export=tmp_path/'export'
    monkeypatch.setattr(sys,'argv',['run.py','--config',str(config_path),'export','--run-dir',str(run),'--output',str(export)])
    cli.main()
    assert file_digest(export/'references/test.npz')==before
    assert json.loads((export/'status.json').read_text())['completed_steps']==['export']
    (run/'gaussian_state.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):verified_fit_bundle(run)
