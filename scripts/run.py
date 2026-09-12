#!/usr/bin/env python3
"""论文主线统一CLI：算法在src，旧实验脚本不是运行依赖。"""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from config import load_paper_config,config_digest
from reference import load_bundle,file_digest
from artifacts import create_paper_run,paper_json,finish_paper_stage,PaperLog,read_paper_scores,resume_paper_run


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def references(config,domains):
    manifest_path=resolve(config['reference']['manifest'])
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='completed':raise ValueError('参考包尚未完成')
    registry={r['dataset']:r for r in manifest['references']}
    result={};hashes={}
    for domain in domains:
        if domain not in registry:raise ValueError(f'没有显式目标参考：{domain}')
        path=resolve(config['reference']['directory'])/f'{domain}.npz'
        digest=file_digest(path)
        if digest!=registry[domain]['sha256']:raise ValueError('参考包哈希不匹配')
        bundle=load_bundle(path)
        if bundle.metadata['dataset']!=domain:raise ValueError('参考包目标域不匹配')
        if config['selection']['name']!=bundle.metadata.get('selector','feature_change') or config['selection']['k']!=bundle.metadata.get('requested_k',3):
            raise ValueError('选择器/K与参考CDF不匹配')
        result[domain]=bundle;hashes[domain]=digest
    return result,hashes


def main():
    import pandas as pd
    from selection import validate_indices
    from workflow import VideoScorer,video_file_states,replay_checkpointed,score_manifest_distributed
    from data.video import video_metadata,downsample_indices
    from evaluation.tables import evaluate_fixed_pairs
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/paper.yaml')
    parser.add_argument('--set',action='append',default=[],help='严格点分覆盖，例如runtime.device=cuda:1')
    commands=parser.add_subparsers(dest='action',required=True)
    study=commands.add_parser('study',help='论文前五组的共享拟合/证据/表格；不调用旧研究脚本')
    study.add_argument('stage',choices=('source-fit','representation-fit','evidence','official','tables','official-tables'))
    study.add_argument('--dataset',required=True)
    study.add_argument('--output',type=Path,required=True)
    study.add_argument('--manifest',type=Path)
    study.add_argument('--fit-run',type=Path)
    study.add_argument('--evidence-run',type=Path)
    study.add_argument('--cdf-run',type=Path)
    study.add_argument('--include-uniform',action='store_true')
    study.add_argument('--sequential-decode',action='store_true',help='共享证据使用已核验的顺序解码，写入实现身份')
    scope=study.add_mutually_exclusive_group()
    scope.add_argument('--baseline-only',action='store_true',dest='baseline_only',help='只评分固定主线')
    scope.add_argument('--all-evidence',action='store_false',dest='baseline_only',help='强制完整表示和统计证据')
    study.set_defaults(baseline_only=None)
    study.add_argument('--rank',type=int,default=0)
    study.add_argument('--world-size',type=int,default=2)
    study.add_argument('--limit',type=int)
    study.add_argument('--dry-run',action='store_true')
    predict=commands.add_parser('predict',help='显式目标参考的单视频推理')
    predict.add_argument('--video',type=Path,required=True)
    predict.add_argument('--dataset',required=True)
    predict.add_argument('--indices-json',type=Path)
    score=commands.add_parser('score',help='按冻结manifest评分，可选固定视频分片双卡')
    score.add_argument('--manifest',type=Path)
    score.add_argument('--dataset',required=True)
    score.add_argument('--limit',type=int,help='明确标为小样本运行，不作主表')
    replay=commands.add_parser('replay',help='匹配参考的标准窗口raw重算，不提DINO')
    replay.add_argument('--window-scores',type=Path,required=True)
    fit=commands.add_parser('fit',help='真实原视频或轻量资产拟合GS/GT/LT，再从独立原视频建立CDF')
    fit.add_argument('--dataset',required=True)
    for role in ('fit','cdf','evaluation','threshold'):
        fit.add_argument(f'--{role}-manifest',type=Path,help='默认使用活跃清单；评测/阈值仅检查隔离')
    fit.add_argument('--stop-after',choices=('gaussian','cdf'),default='cdf',help='gaussian可暂停，恢复到cdf后才有可部署参考包')
    export=commands.add_parser('export',help='核验并导出已完成fit的参考包，不修改默认参考')
    export.add_argument('--run-dir',type=Path,required=True)
    components=commands.add_parser('components',help='冻结分支分数的五项组件移除，不重提特征、不调权重')
    components.add_argument('--run-dir',type=Path,required=True)
    components.add_argument('--pairs',type=Path,required=True)
    for command in (score,replay,fit):command.add_argument('--resume',action='store_true',help='同配置/输入/源码恢复，必须指定--output')
    evaluate=commands.add_parser('evaluate',help='从已完成评分及固定pair_ids评价')
    evaluate.add_argument('--run-dir',type=Path,required=True)
    evaluate.add_argument('--pairs',type=Path,required=True)
    evaluate.add_argument('--resume',action='store_true',help='同评分/配对/源码恢复未完成评价发布')
    bootstrap=commands.add_parser('bootstrap',help='已有候选/基线分数的源组配对区间，不重新计算模型')
    bootstrap.add_argument('--run-dir',type=Path,required=True)
    bootstrap.add_argument('--baseline-run',type=Path,required=True)
    bootstrap.add_argument('--variant',help='候选为components run时必须指定变体')
    bootstrap.add_argument('--baseline-variant',help='基线为components run时必须指定变体')
    bootstrap.add_argument('--pairs',type=Path,required=True)
    for command in (predict,score,replay,bootstrap,fit,export,components):
        command.add_argument('--output',type=Path,help='必须是不存在的新run目录')
    for command in (predict,score,replay,evaluate,bootstrap,fit,export,components):command.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if getattr(args,'resume',False) and args.action!='evaluate' and not args.output:parser.error('--resume必须显式指定原--output目录')
    config=load_paper_config(args.config,args.set)
    if args.action=='study':
        required={'representation-fit':['fit_run'],'evidence':['manifest'],
                  'tables':['evidence_run','cdf_run'],'official-tables':['evidence_run']}.get(args.stage,[])
        for key in required:
            if getattr(args,key) is None:parser.error(f'{args.stage}需要--{key.replace("_","-")}')
        if not 0<=args.rank<args.world_size:parser.error('rank须在[0,world-size)内')
        if args.limit is not None and (args.limit<1 or args.stage!='evidence'):parser.error('limit仅用于evidence正整数小批检查')
        if args.dry_run:
            budget=args.baseline_only if args.baseline_only is not None else (config['selection']['name'],config['selection']['k']) in {('feature_change',1),('uniform',3)}
            print(json.dumps(dict(stage=args.stage,dataset=args.dataset,output=str(args.output),config=config,baseline_only=budget,
                note='规划检查；没有加载GPU或生成完成结果'),ensure_ascii=False,indent=2));return
        from evaluation.study_runner import fit_source_bank,run_shared_evidence
        from evaluation.representations import fit_target_bank
        from evaluation.official_baseline import run_official
        from evaluation.study_tables import aggregate_domain,aggregate_official
        if args.stage=='source-fit':fit_source_bank(ROOT,config,resolve(args.output))
        elif args.stage=='representation-fit':fit_target_bank(ROOT,args.dataset,resolve(args.fit_run),resolve(args.output))
        elif args.stage=='evidence':run_shared_evidence(ROOT,config,args.dataset,resolve(args.manifest),resolve(args.output),
            include_uniform=args.include_uniform,rank=args.rank,world_size=args.world_size,limit=args.limit,sequential_decode=args.sequential_decode,baseline_only=args.baseline_only)
        elif args.stage=='official':run_official(ROOT,config,args.dataset,resolve(args.output),args.rank,args.world_size)
        elif args.stage=='tables':aggregate_domain(ROOT,args.dataset,resolve(args.evidence_run),resolve(args.cdf_run),resolve(args.output))
        elif args.stage=='official-tables':aggregate_official(ROOT,args.dataset,resolve(args.evidence_run),resolve(args.output),args.world_size)
        return
    if args.action=='components':
        from evaluation.tables import component_ablation_tables
        scores,source=read_paper_scores(args.run_dir)
        source_config=load_paper_config(args.run_dir/'resolved_config.yaml')
        canonical=dict(global_enabled=True,local_enabled=True,spatial_weight=.5,global_weight=.5)
        if source_config['method']!=canonical or config['method']!=canonical:raise ValueError('组件分析须以等权完整主线为源，不接受更换权重')
        _,hashes=references(config,sorted(scores.dataset.unique()))
        source_manifest=json.loads((args.run_dir/'run_manifest.json').read_text())
        if source_manifest['inputs'].get('references')!=hashes:raise ValueError('组件源run与当前目标参考身份不符')
        pairs=pd.read_csv(args.pairs,dtype={'video_id':str})
        tables=component_ablation_tables(scores,pairs)
        inputs=dict(source=source,pairs=dict(path=str(args.pairs.resolve()),sha256=file_digest(args.pairs)),references=hashes,
                    scope='fixed video component scores; no refit/selection/threshold fitting; deltas are point estimates')
        if args.dry_run:
            print(json.dumps(inputs,indent=2));print(tables['macro_metrics'].to_string(index=False));return
        directory=create_paper_run(ROOT,config,sys.argv,inputs,args.output)
        with PaperLog(directory):
            try:
                for name,table in tables.items():table.to_csv(directory/(name+'.csv.gz' if name=='video_scores' else name+'.csv'),index=False)
                pairs.to_csv(directory/'pair_ids.csv',index=False)
                paper_json(directory/'variants.json',dict(full='0.5 Global + 0.5 Local',global_only='Global',local_only='Local',
                    without_gs='0.5 GT + 0.5 Local',without_gt='0.5 GS + 0.5 Local',
                    notes='移除后按固定分支规则重新分配权重；不宣称纯粹关闭一项且其余原子权重不变'))
                finish_paper_stage(directory,'components')
                print(tables['macro_metrics'].to_string(index=False))
                print(f'[完成:components] {directory}',flush=True)
            except BaseException as exc:
                paper_json(directory/'status.json',dict(status='failed',stage='components',error=str(exc)));raise
        return
    if args.action=='export':
        from reference_fit import verified_fit_bundle,export_fit_bundle
        bundle_path,provenance=verified_fit_bundle(args.run_dir)
        if args.dry_run:print(json.dumps(provenance,indent=2,ensure_ascii=False));return
        directory=create_paper_run(ROOT,config,sys.argv,provenance,args.output)
        with PaperLog(directory):
            try:
                export_fit_bundle(bundle_path,provenance,directory)
                finish_paper_stage(directory,'export')
                print(f'[完成:export] {directory}/references；显式配置reference.directory/manifest使用',flush=True)
            except BaseException as exc:
                paper_json(directory/'status.json',dict(status='failed',stage='export',error=str(exc)));raise
        return
    if args.action=='bootstrap':
        from data.manifest import load_active_groups
        from evaluation.bootstrap import paired_source_contrast
        candidate,candidate_info=read_paper_scores(args.run_dir.resolve(),args.variant)
        baseline,baseline_info=read_paper_scores(args.baseline_run.resolve(),args.baseline_variant)
        pairs=pd.read_csv(args.pairs)
        groups,group_inputs=load_active_groups(ROOT,resolve(config['data']['manifests']),candidate)
        inputs=dict(candidate=candidate_info,baseline=baseline_info,pairs=dict(path=str(args.pairs.resolve()),sha256=file_digest(args.pairs)),
                    source_manifests=group_inputs,seed_policy='SeedSequence(config seed, stable dataset hash)',scope='compare frozen scores; no rescoring')
        if args.dry_run:print(json.dumps(inputs,indent=2));return
        directory=create_paper_run(ROOT,config,sys.argv,inputs,args.output)
        with PaperLog(directory):
            try:
                paper_json(directory/'status.json',dict(status='running',stage='bootstrap'))
                delta,interval=paired_source_contrast(candidate,baseline,pairs,groups,
                    seed=config['evaluation']['bootstrap_seed'],iterations=config['evaluation']['bootstrap_iterations'],
                    progress=lambda d:print(f'[bootstrap] {d}完成',flush=True))
                delta.to_csv(directory/'comparison_deltas.csv',index=False)
                interval.to_csv(directory/'comparison_bootstrap.csv',index=False)
                pairs.to_csv(directory/'pair_ids.csv',index=False)
                groups.rename('source_group').to_csv(directory/'source_groups.csv')
                finish_paper_stage(directory,'bootstrap')
                print(f'[完成:bootstrap] {directory}',flush=True)
            except BaseException as exc:
                import traceback
                print(traceback.format_exc(),file=sys.stderr)
                paper_json(directory/'status.json',dict(status='failed',stage='bootstrap',error=str(exc)));raise
        return
    if args.action=='evaluate':
        from workflow import evaluate_run
        evaluate_run(ROOT,config,args.run_dir,args.pairs,sys.argv,resume=args.resume,dry_run=args.dry_run)
        return
    inputs={};frame=None;indices=None;fit_frame=None
    if args.action=='fit':
        from reference_fit import prepare_fit_inputs
        base=resolve(config['data']['manifests'])
        paths={role:(getattr(args,f'{role}_manifest') or base/('vatex' if role in ('cdf','threshold') else args.dataset)/f'{role}.csv').resolve()
               for role in ('fit','cdf','evaluation','threshold')}
        fit_frame,frame,inputs=prepare_fit_inputs(ROOT,config,args.dataset,*(paths[r] for r in ('fit','cdf','evaluation','threshold')))
    elif args.action=='replay':
        path=args.window_scores.resolve()
        frame=pd.read_csv(path,float_precision='round_trip',dtype={'video_id':str})
        domains=sorted(frame.dataset.unique())
        inputs['window_scores']=dict(path=str(path),sha256=file_digest(path))
    elif args.action=='score':
        path=args.manifest.resolve() if args.manifest else resolve(config['data']['manifests'])/args.dataset/'evaluation.csv'
        frame=pd.read_csv(path,dtype={'video_id':str})
        required={'video_id','video_path','dataset','subset','source_model','downsample_idxs'}
        if required-set(frame):raise ValueError('评分manifest字段不完整')
        if frame.video_id.duplicated().any() or not frame.dataset.eq(args.dataset).all():raise ValueError('manifest身份重复或目标域不符')
        if not set(frame.subset).issubset({'real','annotated'}):raise ValueError('评分manifest标签无效；无标签单视频使用predict')
        if args.limit is not None:
            if args.limit<1:raise ValueError('limit必须为正')
            frame=frame.head(args.limit)
        if frame.empty:raise ValueError('评分manifest为空')
        for value in frame.downsample_idxs:
            if len(validate_indices(json.loads(value)))<16:raise ValueError('manifest含不足16帧的视频')
        domains=[args.dataset]
        inputs['manifest']=dict(path=str(path),sha256=file_digest(path),selected_videos=len(frame),limit=args.limit)
        inputs['video_files']=video_file_states(ROOT,frame)
    else:
        path=args.video.resolve()
        if not path.is_file():raise FileNotFoundError(path)
        if args.indices_json:
            indices=validate_indices(json.loads(args.indices_json.read_text()))
            inputs['indices']=dict(path=str(args.indices_json.resolve()),sha256=file_digest(args.indices_json))
        else:
            probe=video_metadata(path)
            indices=downsample_indices(probe['num_frames'],probe['fps'])
            inputs['probe']=probe
        if len(indices)<16:raise ValueError('短视频不足16互异帧，不填帧')
        domains=[args.dataset];inputs['video']=dict(path=str(path),sha256=file_digest(path))
    if args.action!='fit':
        bundles,hashes=references(config,domains)
        inputs['references']=hashes
    if args.dry_run:
        if getattr(args,'resume',False):resume_paper_run(ROOT,config,inputs,args.output,args.action)
        brief={key:dict(count=len(value),sha256=config_digest(value)) if key in ('fit_assets','video_files','fit_video_files','cdf_video_files') else value
               for key,value in inputs.items()}
        print(json.dumps(dict(action=args.action,config=config,inputs=brief),indent=2,ensure_ascii=False));return
    if args.action in ('score','predict') and file_digest(resolve(config['encoder']['weights']))!=config['encoder']['weights_sha256']:
        raise ValueError('DINO权重不匹配')
    directory=args.output.resolve() if getattr(args,'resume',False) else create_paper_run(ROOT,config,sys.argv,inputs,args.output)
    lock=(directory/'.run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if getattr(args,'resume',False):
            directory,complete=resume_paper_run(ROOT,config,inputs,directory,args.action)
            if complete:
                print(f'[已经完成:{args.action}] {directory}',flush=True)
                lock.close();return
        manifest=json.loads((directory/'run_manifest.json').read_text())
        manifest['action']=args.action
        manifest.setdefault('stage_inputs',{})[args.action]=inputs
        identity=config_digest(dict(config=manifest['config_sha256'],inputs=inputs,code=manifest['code_sha256']))
        manifest['checkpoint_identity']=identity
        if getattr(args,'resume',False):manifest.setdefault('resume_commands',[]).append(sys.argv)
        paper_json(directory/'run_manifest.json',manifest)
    except BaseException:
        lock.close();raise
    log=PaperLog(directory)
    try:log.__enter__()
    except BaseException:
        lock.close();raise
    try:
        print(f'[开始:{args.action}] {directory}',flush=True)
        paper_json(directory/'status.json',dict(status='running',stage=args.action))
        if args.action=='fit':
            from reference_fit import run_reference_fit
            def progress(stage,index,total,vid):
                paper_json(directory/'progress.json',dict(stage=stage,completed=index,total=total,video_id=vid))
                if index%16==0 or index==total:print(f'[fit:{stage}] {index}/{total}',flush=True)
            result=run_reference_fit(ROOT,config,args.dataset,fit_frame,frame,inputs,directory,identity,
                                     stop_after=args.stop_after,progress=progress)
            if result is None:
                finish_paper_stage(directory,'fit_gaussian')
                paper_json(directory/'status.json',dict(status='paused',stage='fit_gaussian',completed_steps=['fit_gaussian'],
                    next='same command --resume --stop-after cdf; reference bundle not yet deployable'))
                print('[暂停:fit] Gaussian完成，尚未建立CDF，不是可部署参考包',flush=True)
                return
        elif args.action=='replay':
            total=frame.video_id.nunique()
            def progress(index,vid):
                if index%100==0 or index==total:
                    paper_json(directory/'progress.json',dict(completed=index,total=total,video_id=vid))
                    print(f'[replay] {index}/{total}',flush=True)
            scores=replay_checkpointed(frame,bundles,hashes,config['method'],directory,identity,progress)
            scores.to_csv(directory/'video_scores.csv',index=False)
        else:
            selection=dict(selector=config['selection']['name'],k=config['selection']['k'],method=config['method'])
            if args.action=='predict':
                scorer=VideoScorer(bundles[args.dataset],config['runtime']['device'],
                                   dino_repo=str(resolve(config['encoder']['repo'])),dino_weights=str(resolve(config['encoder']['weights'])))
                result=scorer.score(path,indices,**selection)
                paper_json(directory/'prediction.json',dict(video_path=str(path),dataset=args.dataset,reference_sha256=hashes[args.dataset],
                    threshold=None,score_is_probability=False,**result))
            else:
                def progress(index,vid):
                    paper_json(directory/'progress.json',dict(completed=index,total=len(frame),video_id=vid))
                    if index%16==0 or index==len(frame):print(f'[score] {index}/{len(frame)}',flush=True)
                rows,windows=score_manifest_distributed(ROOT,config,frame,bundles[args.dataset],hashes[args.dataset],directory,
                                                        identity,inputs['video_files'],progress)
                if video_file_states(ROOT,frame)!=inputs['video_files']:raise ValueError('评分期间输入视频集合状态改变')
                rows.to_csv(directory/'video_scores.csv',index=False)
                windows.to_csv(directory/'window_scores.csv',index=False)
        finish_paper_stage(directory,args.action)
        print(f'[完成:{args.action}] {directory}',flush=True)
    except BaseException as exc:
        import traceback
        print(traceback.format_exc(),file=sys.stderr)
        paper_json(directory/'status.json',dict(status='failed',stage=args.action,error=str(exc)));raise
    finally:
        log.__exit__(None,None,None)
        lock.close()


if __name__=='__main__':main()
