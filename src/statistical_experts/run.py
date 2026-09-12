"""独立Global专家研究入口；不修改论文默认推理。"""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=['prepare','cache','fit','score','evaluate','analyze','verify','report','diagnose','measure']
    for prefix,actions in {
        'joint':['prepare','score','evaluate','analyze','contrast','verify','report'],
        'predictive':['prepare','fit','score','evaluate','analyze','contrast','verify','report'],
        'controls':['prepare','score','evaluate','analyze','contrast','verify','audit','report'],
        'moments':['score','evaluate','analyze','contrast','verify'],
        'routing':['prepare','score','evaluate','analyze','contrast','verify','report'],
        'source':['prepare','cache','fit','score','evaluate','analyze','contrast','verify','report'],
        'external':['prepare','cache','score','evaluate','analyze','contrast','verify','report'],
    }.items():
        stages.extend(prefix+'-'+action for action in actions)
    p.add_argument('stage',choices=stages)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=2);p.add_argument('--pilot',action='store_true')
    p.add_argument('--length',type=int,choices=[8,16])
    p.add_argument('--contrast',help='后续归因的固定配对对比名')
    a=p.parse_args();root=Path.cwd()
    if a.world_size<1 or not 0<=a.rank<a.world_size:p.error('rank必须在world-size范围内')
    if a.stage.startswith('joint-'):
        from statistical_experts import joint_study
        phase=a.stage.removeprefix('joint-')
        if phase in ('verify','report'):
            from statistical_experts import joint_audit
            getattr(joint_audit,phase)(root)
        elif phase=='score':
            if a.length is None:p.error('联合评分需要--length')
            joint_study.score(root,a.length,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in joint_study.CONTRASTS:p.error('未知联合对比')
            joint_study.contrast(root,a.contrast)
        else:getattr(joint_study,phase)(root)
    elif a.stage.startswith('predictive-'):
        from statistical_experts import predictive
        phase=a.stage.removeprefix('predictive-')
        if phase in ('verify','report'):
            from statistical_experts import predictive_audit
            getattr(predictive_audit,phase)(root)
        elif phase in ('fit','score'):
            if a.length is None:p.error('预测实验需要--length')
            getattr(predictive,phase)(root,a.length,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in predictive.CONTRASTS:p.error('未知预测对比')
            predictive.contrast(root,a.contrast)
        else:getattr(predictive,phase)(root)
    elif a.stage.startswith('external-'):
        from statistical_experts import external_study
        phase=a.stage.removeprefix('external-')
        if phase=='cache':external_study.extract(root,a.rank,a.world_size)
        elif phase=='score':external_study.score(root,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in external_study.CONTRASTS:p.error('未知外部对比')
            external_study.contrast(root,a.contrast)
        elif phase in ('evaluate','verify','report'):
            from statistical_experts import external_audit
            getattr(external_audit,phase)(root)
        else:getattr(external_study,phase)(root)
    elif a.stage.startswith('source-'):
        from statistical_experts import source_data
        phase=a.stage.removeprefix('source-')
        if phase=='prepare':source_data.prepare(root)
        elif phase=='cache':source_data.extract(root,a.rank,a.world_size)
        elif phase in ('verify','report'):
            from statistical_experts import source_audit
            getattr(source_audit,phase)(root)
        else:
            from statistical_experts import source_study
            if phase in ('fit','score'):
                if a.length is None:p.error('来源实验需要--length')
                getattr(source_study,phase)(root,a.length,f'cuda:{a.rank%2}')
            elif phase=='contrast':
                if a.contrast not in source_study.CONTRASTS:p.error('未知来源对比')
                source_study.contrast(root,a.contrast)
            else:getattr(source_study,phase)(root)
    elif a.stage.startswith('routing-'):
        from statistical_experts import routing_study
        phase=a.stage.removeprefix('routing-')
        if phase in ('verify','report'):
            from statistical_experts import routing_audit
            getattr(routing_audit,phase)(root)
        elif phase=='score':
            if a.length is None:p.error('连续度量评分需要--length')
            routing_study.score(root,a.length,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in routing_study.CONTRASTS:p.error('未知路由对比')
            routing_study.contrast(root,a.contrast)
        else:getattr(routing_study,phase)(root)
    elif a.stage.startswith('moments-'):
        from statistical_experts import moment_controls
        phase=a.stage.removeprefix('moments-')
        if phase=='score':
            if a.length is None:p.error('四格评分需要--length')
            moment_controls.score(root,a.length,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in moment_controls.CONTRASTS:p.error('未知四格对比')
            moment_controls.contrast(root,a.contrast)
        else:getattr(moment_controls,phase)(root)
    elif a.stage.startswith('controls-'):
        from statistical_experts import controls
        phase=a.stage.removeprefix('controls-')
        if phase=='report':
            from statistical_experts.controls_report import report
            report(root)
        elif phase=='audit':
            from statistical_experts.controls_audit import audit
            audit(root)
        elif phase=='score':
            if a.length is None:p.error('CDF评分需要--length')
            controls.score_reference(root,a.length,f'cuda:{a.rank%2}')
        elif phase=='contrast':
            if a.contrast not in controls.CONTRASTS:p.error('未知配对对比')
            controls.contrast(root,a.contrast)
        else:getattr(controls,phase)(root)
    elif a.stage=='prepare':
        from statistical_experts.manifests import prepare
        prepare(root)
    elif a.stage=='cache':
        from statistical_experts.cache import extract
        extract(root,a.rank,a.world_size,pilot=a.pilot)
    elif a.stage=='fit':
        from statistical_experts.engine import fit_models
        if a.length is None:p.error('拟合需要--length')
        fit_models(root,a.length,f'cuda:{a.rank%2}')
    elif a.stage=='score':
        from statistical_experts.scoring import score
        if a.length is None:p.error('评分需要--length')
        score(root,a.length,a.rank,a.world_size,pilot=a.pilot)
    elif a.stage=='evaluate':
        from statistical_experts.evaluation import evaluate
        evaluate(root)
    elif a.stage=='analyze':
        from statistical_experts.analysis import all_contrasts
        all_contrasts(root)
    elif a.stage=='verify':
        from statistical_experts.verify import verify
        verify(root)
    elif a.stage=='report':
        from statistical_experts.report import report
        report(root)
    elif a.stage=='diagnose':
        from statistical_experts.diagnostics import real_holdout
        real_holdout(root)
    elif a.stage=='measure':
        from statistical_experts.measure import measure
        measure(root)


if __name__=='__main__':main()
