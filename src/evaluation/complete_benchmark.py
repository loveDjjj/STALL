"""补齐23单元：共享身份合并8/16帧结果，保留原文动态筛选的独立协议。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import checkpoint_read,paper_json
from config import config_digest
from reference import file_digest
from evaluation.study_tables import read_evidence,calibrated_scores
from evaluation.tables import evaluate_fixed_pairs
from evaluation.paper_export import STUDIES


def checked_scores(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('输入表未完成')
    path=directory/'video_scores.csv.gz'
    if file_digest(path)!=manifest['files'][path.name]:raise ValueError('源分数hash改变')
    return pd.read_csv(path,float_precision='round_trip')


def official_short(root,domain,frame):
    directory=root/'results/runs'/f'complete23_{domain}_official_short8'
    spec=json.loads((directory/'identity_rank_0.json').read_text())
    done=json.loads((directory/'completed_rank_0.json').read_text())
    identity=config_digest(spec)
    if done['identity']!=identity or spec['manifest_sha256']!=config_digest(frame.to_dict('records')):
        raise ValueError('官方短视频结果身份不符')
    if done['world_size']!=1 or spec['window_frames']!=8:raise ValueError('官方短视频分片/帧数不符')
    rows=[]
    for i,r in enumerate(frame.to_dict('records')):
        score=checkpoint_read(directory/'raw'/f'{i:06d}.json',identity,r['video_id'])
        if score['frame_indices']!=json.loads(r['downsample_idxs']):raise ValueError('官方1秒帧身份不符')
        rows.append({k:r[k] for k in ('video_id','dataset','subset','source_model','video_path')}|
                    dict(variant='official_single_window',final_score=score['final_score'],effective_k=1))
    return pd.DataFrame(rows)


def prepare_short_tables(root,domain):
    root=Path(root);output=root/'results/runs'/f'complete23_tables_{domain}_short8'
    if output.exists():return checked_scores(output)
    manifest=root/'data/manifests/short_video'/domain/'evaluation.csv'
    frame=pd.read_csv(manifest,keep_default_na=False)
    cframe=pd.read_csv(root/'data/manifests/short_video/vatex/cdf.csv',keep_default_na=False)
    ev,ei=read_evidence(root/'results/runs'/f'complete23_{domain}_evaluation_short8',frame)
    cdf,ci=read_evidence(root/'results/runs'/f'complete23_{domain}_cdf_short8',cframe)
    if ei['models']!=ci['models'] or ei['config']!=ci['config']:raise ValueError('8帧评价和CDF参数不同')
    if ei['config']['evidence_window_frames']!=8:raise ValueError('短视频参考不是8帧')
    if any(len(r['windows'])!=1 or len(r['windows'][0]['frame_indices'])!=8 for r in ev):
        raise ValueError('短视频观察预算不同')
    scores=calibrated_scores(root,domain,frame,ev,cdf,1)
    scores=pd.concat([scores,official_short(root,domain,frame)],ignore_index=True)
    pairs=pd.read_csv(root/'data/manifests/short_video'/domain/'pairs.csv')
    output.mkdir(parents=True,exist_ok=False);scores.to_csv(output/'video_scores.csv.gz',index=False)
    tables=[]
    for variant,part in scores.groupby('variant'):
        tables.append(evaluate_fixed_pairs(part,pairs)['generator_metrics'].assign(variant=variant))
    pd.concat(tables).to_csv(output/'generator_metrics.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',models=ei['models'],selection=ei['config']['selection'],
        evaluation_manifest_sha256=file_digest(manifest),files={p.name:file_digest(p) for p in output.iterdir() if p.is_file()}))
    return scores


def combine(root,output,scope,native_official=None):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise ValueError('完整协议导出不可覆盖')
    short={d:prepare_short_tables(root,d) for d in ('genvideo','videofeedback')}
    long_pairs=pd.read_csv(root/'data/manifests/active/pairs.csv',keep_default_na=False)
    long_pairs=long_pairs[long_pairs.dataset.isin(['comgenvid','genvideo','videofeedback'])]
    if scope in ('paper_filter','paper_aligned'):
        audit='complete23_official_pairing_audit' if scope=='paper_aligned' else 'videofeedback_dynamic_filter_audit'
        filtered=pd.read_csv(root/'results/runs'/audit/'filtered_pairs.csv',keep_default_na=False)
        long_pairs=pd.concat([long_pairs[long_pairs.dataset!='videofeedback'],filtered],ignore_index=True)
    pairs=pd.concat([long_pairs,*[pd.read_csv(root/'data/manifests/short_video'/d/'pairs.csv',keep_default_na=False) for d in short]],ignore_index=True)
    cells=set(map(tuple,pairs[['dataset','generator']].drop_duplicates().to_numpy()))
    if len(cells)!=23:raise ValueError('完整协议不是23单元')
    manifests=[]
    for d in ('comgenvid','genvideo','videofeedback'):
        manifests.append(pd.read_csv(root/'data/manifests/active'/d/'evaluation.csv',keep_default_na=False))
        if d in short:manifests.append(pd.read_csv(root/'data/manifests/short_video'/d/'evaluation.csv',keep_default_na=False))
    frame=pd.concat(manifests,ignore_index=True).fillna('')
    frame=frame[frame.video_id.isin(set(pairs.video_id))]
    if frame.video_id.duplicated().any():raise ValueError('跨时长评价身份重复')
    modes={};source_hashes={}
    for mode in ('official','fc3','uniform1','fc1','uniform3'):
        source=Path(native_official) if mode=='official' and native_official else root/'results/runs'/f'paper_tables_all_{mode}'
        long=checked_scores(source);source_hashes[str(source)]=file_digest(source/'manifest.json')
        variants=set(long.variant)
        parts=[long] if mode=='official' and native_official else [long,*[s[s.variant.isin(variants)] for s in short.values()]]
        combined=pd.concat(parts,ignore_index=True);combined=combined[combined.video_id.isin(set(pairs.video_id))]
        for variant,part in combined.groupby('variant'):
            if part.video_id.duplicated().any() or set(part.video_id)!=set(pairs.video_id):raise ValueError(f'{mode}/{variant}身份不完整')
        modes[mode]=combined
    output.mkdir(parents=True,exist_ok=False)
    pairs.to_csv(output/'pairs.csv',index=False);frame.to_csv(output/'evaluation.csv',index=False)
    for mode,scores in modes.items():scores.to_csv(output/f'scores_{mode}.csv.gz',index=False)
    for study,definitions in STUDIES.items():
        cells_out=[];summary=[]
        for mode,variant,label in definitions:
            part=modes[mode];part=part[part.variant==variant]
            tables=evaluate_fixed_pairs(part,pairs);g=tables['generator_metrics']
            if set(map(tuple,g[['dataset','generator']].to_numpy()))!=cells:raise ValueError('消融未覆盖23单元')
            cells_out.append(g.assign(variant=variant,method=label,observation=mode))
            dataset=tables['dataset_metrics'];macro=tables['macro_metrics'].rename(columns={'scope':'dataset'})
            metrics=[c for c in dataset if c!='dataset']
            average=pd.DataFrame([dict(dataset='Average',**g[metrics].mean().to_dict())])
            summary.append(pd.concat([dataset,average,macro]).assign(variant=variant,method=label,observation=mode))
        pd.concat(cells_out).to_csv(output/f'{study}_generators.csv',index=False)
        pd.concat(summary).to_csv(output/f'{study}.csv',index=False)
    paper_json(output/'manifest.json',dict(status='point_estimates_complete',scope=scope,cells=23,
        official_native_source=str(Path(native_official).resolve()) if native_official else None,
        unique_evaluation=len(frame),pair_rows=len(pairs),source_manifests=source_hashes,
        short_sources={d:file_digest(root/'results/runs'/f'complete23_tables_{d}_short8/manifest.json') for d in short},
        note='短视频在四个观察策略上是相同8帧/单窗；动态过滤仅由官方元数据决定，不按检测结果选择',
        files={p.name:file_digest(p) for p in output.iterdir() if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['coverage_only','paper_filter','paper_aligned'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--native-official')
    args=parser.parse_args();combine(Path.cwd(),args.output,args.scope,args.native_official)
