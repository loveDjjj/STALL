"""训练支持与生成器名称重叠的只读统计，不接触预测选择。"""
from pathlib import Path
import pandas as pd
import numpy as np
from artifacts import atomic_csv,paper_json
from reference import file_digest
from discriminative_moe.run import configuration
from discriminative_moe.training import sampling_weights,TRAINING_PROTOCOL


def audit_design(root):
    root=Path(root);out=root/configuration(root)['run_directory'];roles=pd.read_csv(out/'roles.csv');meta=pd.read_csv(out/'videos.csv')
    support=[];overlap=[]
    for fold,assignment in roles.groupby('fold'):
        train=meta[meta.video_id.isin(assignment[assignment.role=='train'].video_id)].reset_index(drop=True)
        weights=sampling_weights(train);train['sampling_mass']=weights
        for (domain,length,label),q in train.groupby(['dataset','length','subset']):
            mass=q.groupby('split_group').sampling_mass.sum().to_numpy()
            support.append(dict(fold=fold,dataset=domain,length=length,subset=label,clips=len(q),source_groups=q.split_group.nunique(),
                sampling_mass=mass.sum(),effective_source_count=mass.sum()**2/(mass*mass).sum(),maximum_source_mass=mass.max()))
        test=meta[meta.video_id.isin(assignment[assignment.role=='test'].video_id)]
        seen=set(train.loc[train.subset.eq('annotated'),'source_model'])
        for (domain,generator),q in test[test.subset.eq('annotated')].groupby(['dataset','source_model']):
            overlap.append(dict(fold=fold,dataset=domain,test_generator=generator,clips=len(q),
                exact_name_seen_in_training=generator in seen,
                casefold_name_seen_in_training=generator.casefold() in {x.casefold() for x in seen},
                family_independence='not_established_by_name_check'))
    atomic_csv(out/'training_support.csv',pd.DataFrame(support));atomic_csv(out/'generator_name_overlap.csv',pd.DataFrame(overlap))
    paper_json(out/'design_audit.json',dict(status='verified',training_protocol=TRAINING_PROTOCOL,meaning='域/长度/标签权重支持与名称重叠；不声称已完成生成器家族排除',
        files={p:file_digest(out/p) for p in ['training_support.csv','generator_name_overlap.csv']}))
    print(pd.DataFrame(support).to_string(index=False),flush=True)


if __name__=='__main__':audit_design(Path(__file__).resolve().parents[2])
