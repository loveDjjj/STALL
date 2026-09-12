"""容量与预热诊断的最终表格和边界。"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from moe_training.run import configuration


def report(root):
    c=configuration(root);out=root/c['run_directory'];v=json.loads((out/'verification.json').read_text())
    if v['status']!='verified':raise ValueError('尚未通过验收')
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('测试未通过')
    test_count=sum(int(s.attrib['tests']) for s in suites)
    macro=pd.read_csv(out/'macro_metrics.csv');ds=pd.read_csv(out/'dataset_metrics.csv');ci=pd.read_csv(out/'confidence_intervals.csv');sel=pd.read_csv(out/'selected_epochs.csv')
    labels={'narrow_mlp':'旧宽128 MLP','narrow_uniform':'旧2×64均匀头','narrow_moe':'旧2×64 MoE',
            'wide_mlp':'宽256 MLP','uniform':'2×128均匀头','joint':'2×128联合MoE','warm':'2×128预热MoE'}
    lines=['# Global MoE：容量与训练日程诊断','',
        '本轮固定Global摘要、源级划分、长度匹配与三seed。宽MLP256；双头每个128。预热组前10轮gate固定为均匀，第11轮解冻，共30轮，不增加训练预算。旧窄模型单列作为容量参照；原论文主线不变。',
        '', '| 模型 | 开发Average AUC/AP-real | GenVidBench AUC/AP-real | ViF AUC/AP-real |','| --- | ---: | ---: | ---: |']
    for head,label in labels.items():
        q=macro[macro['head']==head][['auc','real_positive_ap']].mean();values=[f'{q.auc:.6f}/{q.real_positive_ap:.6f}']
        for domain in ['genvidbench','vifbench']:
            q=ds[(ds.scope_kind=='external')&(ds['head']==head)&ds.dataset.eq(domain)][['auc','real_positive_ap']].mean();values.append(f'{q.auc:.6f}/{q.real_positive_ap:.6f}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '上述为三个seed指标均值，不是预测平均后的ensemble；开发按域内生成器、域间等权。全部seed和逐生成器AP-fake、ROC操作点保存在CSV。外部已经多轮观察，不称新盲测，GenVidBench仅一个ms单元。',
        '', '## 开发配对区间','', '| 对比 | ΔAUC [95%CI] | ΔAP-real [95%CI] |','| --- | ---: | ---: |']
    for name,q in ci[(ci.scope_kind=='development')&ci.dataset.eq('Average')].groupby('contrast',sort=False):
        values=[]
        for metric in ['auc','ap_real']:
            r=q[q.metric==metric].iloc[0];values.append(f'{r.delta:+.6f} [{r.ci95_low:+.6f},{r.ci95_high:+.6f}]')
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '区间是固定模型、固定划分下1000次源级Poisson重抽样；同一源的权重在三个seed间共享，未覆盖全部研究选择不确定性。',
        '', '## 训练机制','', '| fold | seed | warm最佳epoch | 是否尚未启用gate |','| --- | ---: | ---: | --- |']
    for r in sel[sel['mode']=='warm'].itertuples():lines.append(f'| {r.fold} | {r.seed} | {r.epoch} | {r.selected_before_gate_enabled} |')
    lines += ['', '若warm选中了前10轮，最终模型实际上仍是均匀路由，不能将它的成绩算作学习gate的成功。每epoch history.csv包含抽样hash、updates、gate质量/熵与裁剪前专家/路由梯度；expert_validation.csv包含每专家按内容簇、来源和类别的NLL。内容簇只由训练Global拟合，不进入训练目标，不能据其编号宣称人物/动物专家。',
        '', f'验收：{test_count}项tests通过，{v["models"]}个模型，warm与uniform的第10轮专家参数逐元素一致；四组抽样及updates一致。独立NumPy探针{v["independent_validation_probes"]}个，最大概率误差{v["probability_error"]:.3g}。全部生成器指标重新核算通过。',
        '', '判定需区分：宽模型共同提高属于容量作用；warm优于joint只说明训练日程有帮助；warm还需超过uniform和宽MLP才支持保留路由。任何不利域或低FPR退化均保留，不按域拼模型。不自动启动内容专家、更多容量或warmup扫描。','']
    lines += ['## 本轮结论','',
        '扩大专家后joint相对旧窄MoE的开发AUC/AP增益区间均为正，但相对同容量uniform的两项区间均跨零。固定10轮预热相对joint及uniform均无明确增益。这支持“此前容量约束影响了MoE”，不支持“只是训练没收敛/预热一下就能胜过简单组合”。',
        '', 'pooled三个warm最佳epoch分别6、7、4，与uniform相同，全部位于均匀预热阶段；它们的外部结果相同是预期退化边界，不是两个算法恰好表现相近。开发warm中仍有选择解冻后模型的fold，全部保留。',
        '', '保留旧Global均匀双头作为既有监督参照，不按数据域切换模型，也不替换原real-only论文主线。本轮完成，停止继续扫描warmup、容量或gate温度。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    source=root/c['source_directory']
    paper_json(out/'decision.json',dict(status='completed_no_stable_routing_gain',retain_previous_baseline=True,
        capacity_effect='development gain for joint over narrow MoE; not a consistent external gain',
        warmup_effect='no clear gain over joint or uniform; pooled selected uniform phase',
        intervals=file_digest(out/'confidence_intervals.csv')))
    paper_json(out/'manifest.json',dict(status='completed',config=c,tests=test_count,verification=file_digest(out/'verification.json'),
        source_inputs={str(p.relative_to(root)):file_digest(p) for p in [root/'results/paper_complete/pairs.csv',source/'test_scores.csv.gz',
            source/'external/test_scores.csv.gz',source/'external/evaluation_manifest.json',source/'external/pairs.csv']},
        files={p.name:file_digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
