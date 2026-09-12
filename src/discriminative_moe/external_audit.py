"""外部冻结分类器与指标验收；不据外部表现修改模型。"""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from artifacts import paper_json
from reference import file_digest
from config import config_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.run import ROOT,configuration
from discriminative_moe.audit import independent_probability


def verify(root=ROOT):
    root=Path(root);c=configuration(root);parent=root/c['run_directory'];out=parent/'external'
    receipt=json.loads((out/'evaluation_manifest.json').read_text());prepared=json.loads((out/'prepared.json').read_text())
    assert receipt['selection']==file_digest(parent/'selection.json')==prepared['identity']['selection']
    for manifest in (receipt,prepared):
        for name,h in manifest['files'].items():assert file_digest(out/name)==h
    for name,h in prepared['identity']['inputs'].items():assert file_digest(root/name)==h
    spec=json.loads((out/'features/identity.json').read_text());identity=config_digest(spec)
    for name,h in spec['code'].items():assert file_digest(root/name)==h
    assert spec['prepared']==file_digest(out/'prepared.json')
    meta=pd.read_csv(out/'videos.csv');jobs=json.loads((out/'jobs.json').read_text())['jobs'];assert len(meta)==len(jobs)==2216
    arrays={k:[] for k in ['G','T','L']};mask=[]
    for j in jobs:
        p=out/'features'/(j['key']+'.npz');assert file_digest(p)==receipt['feature_hashes'][j['key']]
        with np.load(p) as z:
            assert str(z['identity'])==identity and str(z['video_id'])==j['video_id']
            k=len(j['windows']);mask.append(np.arange(3)<k)
            for name,d in [('G',2048),('T',4064),('L',4064)]:
                assert z[name].shape==(k,d) and np.isfinite(z[name]).all()
                a=np.zeros((3,d),np.float32);a[:k]=z[name];arrays[name].append(a)
    arrays={k:np.stack(v) for k,v in arrays.items()};mask=np.stack(mask)
    np.testing.assert_array_equal([j['video_id'] for j in jobs],meta.video_id)
    recovery=json.loads((out/'infinity_parser_recovery.json').read_text());assert recovery['arrays_exactly_preserved']
    assert recovery['new_identity']==identity
    raw_error=0.;processed=len(recovery['files'])
    for rank in (0,1):
        r=json.loads((out/f'features/completed_{rank}.json').read_text());assert r['identity']==identity
        raw_error=max(raw_error,r['max_global_raw_error']);processed+=r['jobs']
    assert processed==2216 and raw_error<=1e-8
    scores=pd.read_csv(out/'test_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(out/'pairs.csv')
    chosen=pd.read_csv(parent/'selected_capacities.csv');chosen=chosen[chosen.fold=='pooled'];max_error=0.;probes=0
    for (inp,head,seed),q in scores.groupby(['input','head','seed']):
        assert len(q)==q.video_id.nunique()==2216 and np.isfinite(q.final_score).all()
        k='uniform' if head=='uniform_matched' else head
        capacity_kind='moe' if head=='uniform_matched' else head
        n=int(chosen[(chosen.input==inp)&chosen.kind.eq(capacity_kind)].experts.iloc[0])
        key=f'pooled__{inp}__{k}{n}__s{seed}';dest=parent/'training'/key
        manifest=json.loads((dest/'manifest.json').read_text());assert file_digest(dest/'model.pt')==manifest['files']['model.pt']
        state=torch.load(dest/'model.pt',map_location='cpu',weights_only=True);assert state['identity']==manifest['identity']
        names={'G':['G'],'GT':['G','T'],'GTL':['G','T','L']}[inp];x=np.concatenate([arrays[k] for k in names],axis=-1)
        for domain in ['genvidbench','vifbench']:
            for label in ['real','annotated']:
                ix=np.flatnonzero(meta.dataset.eq(domain)&meta.subset.eq(label))[[0,-1]]
                expected=independent_probability(state,x[ix],mask[ix]);actual=1-q.set_index('video_id').loc[meta.iloc[ix].video_id,'final_score'].to_numpy()
                np.testing.assert_allclose(expected,actual,rtol=0,atol=2e-5);max_error=max(max_error,float(np.abs(expected-actual).max()));probes+=len(ix)
        tables=evaluate_fixed_pairs(q,pairs)
        for table,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('full_population_metrics',['dataset'])]:
            a=tables[table].set_index(keys).sort_index();b=pd.read_csv(out/(table+'.csv'),float_precision='round_trip')
            b=b[(b.input==inp)&(b['head']==head)&(b.seed==seed)].drop(columns=['input','head','seed']).set_index(keys).sort_index()
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    assert scores.groupby(['input','head','seed']).ngroups==45
    ds=pd.read_csv(out/'dataset_metrics.csv');ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==24
    for r in ci.itertuples():
        inp,control=r.contrast.split('_moe_vs_');column='auc' if r.metric=='auc' else 'real_positive_ap'
        q=ds[(ds.input==inp)&ds.dataset.eq(r.dataset)]
        expected=q[q['head']=='moe'][column].mean()-q[q['head']==control][column].mean()
        np.testing.assert_allclose(r.delta,expected,rtol=0,atol=1e-12)
    paper_json(out/'verification.json',dict(status='verified',clips=2216,cells=20,configurations=45,
        independent_forward_probes=probes,maximum_probability_error=max_error,global_raw_regression_error=raw_error,
        no_external_training=True,selection=file_digest(parent/'selection.json'),code=file_digest(Path(__file__))))
    print('external verification passed',probes,max_error,flush=True)


if __name__=='__main__':verify()
