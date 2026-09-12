"""监督MoE结果：容量选择、三类证据、跨域与路由分开报告。"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from discriminative_moe.run import configuration
from discriminative_moe.training import TRAINING_PROTOCOL


def report(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];verification=json.loads((out/'verification.json').read_text())
    if verification['status']!='verified':raise ValueError('未通过独立验收')
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('测试未通过')
    tests=sum(int(s.attrib['tests']) for s in suites)
    mean=pd.read_csv(out/'seed_summary.csv');ci=pd.read_csv(out/'confidence_intervals.csv');counts=pd.read_csv(out/'selected_capacities.csv')
    domains=pd.read_csv(out/'dataset_metrics.csv');macro=pd.read_csv(out/'macro_metrics.csv');qualified=[]
    for inp in ('G','GT','GTL'):
        accepted=[]
        for metric,column in [('auc','auc'),('ap_real','real_positive_ap')]:
            reliable=True
            for control in ('mlp','uniform_matched'):
                name=f'{inp}_moe_vs_{control}'
                q=ci[(ci.dataset=='Average')&(ci.contrast==name)]
                own=q[q.metric==metric].iloc[0];other=q[q.metric!=metric].iloc[0]
                reliable &= own.ci95_low>0 and other.ci95_high>=0
                a=macro[(macro.input==inp)&(macro['head']=='moe')].set_index('seed')[column]
                b=macro[(macro.input==inp)&(macro['head']==control)].set_index('seed')[column]
                reliable &= bool((a-b).gt(0).all())
            if reliable:accepted.append(metric)
        if accepted:qualified.append(dict(input=inp,metrics=accepted))
    decision=dict(status='development_requires_external' if qualified else 'development_no_stable_moe_gain',qualified_inputs=qualified,
        rule='两种控制均至少同一主指标源级区间为正，另一指标无明确损害，三个seed点差均正；仍需检查域间代价和外部确认。',
        external_run=False,baseline_unchanged=True,evidence=file_digest(out/'confidence_intervals.csv'))
    labels={'linear':'线性','mlp':'单MLP','uniform':'均匀组合（验证选容量）','uniform_matched':'均匀组合（匹配MoE容量）','moe':'学习路由MoE'}
    lines=['# 冻结DINO的监督证据MoE：开发留域结果','',
        '这是新监督协议：训练real/fake拟合分类头，DINO保持冻结，无Gaussian/CDF。不同于旧主线真实参考方法，不能称training-free。外部数据此前已被研究过，后续若确认也不称首次未接触盲测。','',
        '## 数据与学习目标','',
        '15569个clip-ID、14366连接源组、23个生成器单元；每fold两个域训练/验证，第三域测试。源组、物理同文件及短长裁剪共同划分。源级验证占训练侧约20%，所有标准化只用训练行；真实来源/生成器分层。不把重复配对real展开为训练数据。训练采用length_matched_supervision_v2：域和真假等权、fake生成器等权，real的8/16帧采样质量匹配fake，并在长度层内按源平衡；验证真假匹配帧数。',
        '', '训练支持见training_support.csv，源权重有效数量仅为权重集中程度，不是独立性保证。generator_name_overlap.csv显示Sora、ModelScope等名称跨训练/测试域重叠；名称不同也不证明生成器家族不同，不将本轮留域结果称为严格未见家族。',
        '', 'G为外观一/二阶通道矩；T为归一化Global T1；L为归一化Local D2。T/L额外含固定64投影的非对角二阶矩，不等价于完整1024协方差，也不是新的可训练编码器。三输入组合全部测试，不能用G的负结果替代Local检验。',
        '', '模型按视频级BCE训练；每窗分类概率再视频均值。学习router的MoE为2/4软专家，所有专家均参与计算，各自都是真假分类器。总隐藏宽度128；单MLP与MoE近似参数匹配，router参数另计。专家数根据训练侧验证三seed均值与SE选择，不根据测试选epoch/seed/容量。','',
        '## Average与训练波动','',
        '| 输入 | 分类头 | AUC（3seed均值±标准差） | AP-real（均值±标准差） |','| --- | --- | ---: | ---: |']
    for r in mean.itertuples():
        lines.append(f'| {r.input} | {labels[r.head]} | {r.auc_mean:.6f} ± {r.auc_std:.6f} | {r.real_positive_ap_mean:.6f} ± {r.real_positive_ap_std:.6f} |')
    lines += ['', '先每生成器计算，再域内、三个留出域等权平均；此处是各seed指标平均，不是三seed预测平均后的ensemble指标。原主线0.874472使用额外目标real统计且属于另协议，不作为MoE机制归因基线。','',
        '## 各域结果','', '| 输入 | 分类头 | ComGenVid AUC/AP | GenVideo AUC/AP | VideoFeedback AUC/AP |','| --- | --- | ---: | ---: | ---: |']
    for inp in ('G','GT','GTL'):
        for head in ('linear','mlp','uniform','uniform_matched','moe'):
            vals=[]
            for domain in ('comgenvid','genvideo','videofeedback'):
                q=domains[(domains.input==inp)&(domains['head']==head)&(domains.dataset==domain)][['auc','real_positive_ap']].mean()
                vals.append(f'{q.auc:.6f}/{q.real_positive_ap:.6f}')
            lines.append('| '+inp+' | '+labels[head]+' | '+' | '.join(vals)+' |')
    lines += ['', '各域参数是该fold训练侧验证选定，不是按测试域表现挑出的最优模型。跨域同生成器可能存在，所以这些表不自动代表生成器家族完全未见。',
        '', '## 专家数量选择','', '| 输入 | 留出/用途 | 均匀组合数量 | MoE数量 |','| --- | --- | ---: | ---: |']
    for (inp,fold),q in counts.groupby(['input','fold']):
        q=q.set_index('kind');lines.append(f'| {inp} | {fold} | {int(q.loc["uniform","experts"])} | {int(q.loc["moe","experts"])} |')
    lines += ['', 'pooled仅为后续外部模型选择，没有开发测试输出。训练容量/epoch全表见validation_model_grid.csv；选择标准误来自三个训练seed，不是完整参考重抽样不确定性。',
        '', '## 三seed共享源组的配对区间','', '| 对比 | ΔAUC [95%CI] | ΔAP-real [95%CI] |','| --- | ---: | ---: |']
    for name,q in ci[ci.dataset.eq('Average')].groupby('contrast',sort=False):
        vals=[]
        for metric in ('auc','ap_real'):
            r=q[q.metric.eq(metric)].iloc[0];vals.append(f'{r.delta:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '1000次源组Poisson权重在生成器、候选/控制和三个seed间共同使用，每次计算各seed指标差再平均。区间条件于这三个已训练模型与当前划分，不包含训练池重新抽样，也未调整全部探索的多重比较。',
        '', '## 路由与解释边界','',
        'route_summary.csv按域、真假、来源/生成器记录router质量、平均窗口有效专家数与总体有效专家数，并列出各专家输出均值和标准差；expert_correlations.csv记录专家输出相关性。均匀路由不代表专家有用，域相关路由也不单独证明使用了捷径。若专家输出接近常数而router区分类别，应收窄“不同取证规律分工”的解释。routing_intervention_scores.csv.gz仅为测试时改均匀权重的干预诊断，不替代独立训练的均匀组合。需结合跨域性能、同容量均匀控制与独立确认。',
        '', '## 决定与验收','',
        f'当前决定：{decision["status"]}。达到上述开发门槛的输入：{qualified}。',
        '如果有合格输入，继续预定外部、家族排除及一种统一处理控制；未做这些之前不替换论文主线。没有合格输入则停止当前MoE扩张，保留普通分类器和输入证据的正负发现。',
        f'{tests}项tests通过；{verification["models"]}个模型的训练身份/标准化、{verification["independent_forward_probes"]}个NumPy独立前向探针核验。独立概率最大误差{verification["independent_probability_error"]:.3g}。',
        '', '逐生成器AUC/AP-real/AP-fake及ROC操作点：generator_metrics.csv；全部逐视频：test_scores.csv.gz；专家路由：test_routes.csv.gz；源级用途：roles.csv。ROC操作点不等于独立部署阈值保证。','']
    external=out/'external'
    if (external/'verification.json').exists():
        ev=json.loads((external/'verification.json').read_text())
        if ev['status']!='verified':raise ValueError('外部尚未验收')
        em=pd.read_csv(external/'seed_summary.csv');ed=pd.read_csv(external/'dataset_metrics.csv');ec=pd.read_csv(external/'confidence_intervals.csv')
        lines += ['## 冻结pooled模型的外部核对','',
            '开发MoE未过门槛后，因简单监督头的较高开发结果，追加一次有限的外部核对；该调整在外部评分前记录。全部已有pooled模型冻结，无额外训练或外部选参。不是因某个外部域较好重新开启专家搜索。',
            '', '| 输入 | 分类头 | GenVidBench AUC/AP-real | ViF AUC/AP-real |','| --- | --- | ---: | ---: |']
        for inp in ('G','GT','GTL'):
            for head in ('linear','mlp','uniform','moe','uniform_matched'):
                vals=[]
                for domain in ('genvidbench','vifbench'):
                    r=em[(em.input==inp)&(em['head']==head)&em.dataset.eq(domain)].iloc[0]
                    vals.append(f'{r.auc_mean:.6f}/{r.real_positive_ap_mean:.6f}')
                lines.append('| '+inp+' | '+labels[head]+' | '+' | '.join(vals)+' |')
        lines += ['', 'GenVidBench仅1个ms生成器单元；ViF19单元、85个real文件/83源组。外部不并入开发Average。外部与开发已有物理文件隔离检查，但不是完整语义去重或所有生成器家族独立性证明。',
            '', '| Global输入分类头 | GenVidBench Recall@1%FPR | ViF Recall@1%FPR |','| --- | ---: | ---: |']
        for head in ('linear','mlp','uniform','moe'):
            vals=[ed[(ed.input=='G')&(ed['head']==head)&ed.dataset.eq(d)].fake_tpr_at_1pct_real_fpr.mean() for d in ('genvidbench','vifbench')]
            lines.append(f'| {labels[head]} | {100*vals[0]:.3f}% | {100*vals[1]:.3f}% |')
        lines += ['', 'MoE在ViF的低FPR召回点估计高于均匀双头，不能隐去；但AUC/其他域并未一致改善，不按这个后见操作点改变主门槛。ViF每单元real不足100，1%FPR约束对应离散ROC的零误报档，不能当成独立部署阈值保证。',
            '', f'外部{ev["clips"]}身份/{ev["cells"]}单元、{ev["configurations"]}配置（含3seed）与{ev["independent_forward_probes"]}个独立NumPy预测探针通过；最大概率误差{ev["maximum_probability_error"]:.3g}，Global raw回归误差{ev["global_raw_regression_error"]:.3g}。',
            '', '外部完整CSV及区间位于external/。静态GT的JSON Infinity读取问题已单行修复，559个已存摘要逐元素不变恢复，记录external/infinity_parser_recovery.json；没有丢弃静态视频或扩大误差容限。',
            '', '## 研究结论','',
            '冻结Global特征的监督判别已提供明显信号，但条件路由MoE没有稳定超过普通MLP和均匀组合。当前值得保留的监督参照是Global输入的简单分类头/均匀双头；2专家是G输入各fold及pooled验证选择的数量，不是根据ViF挑选。不得按数据域在MLP、均匀、MoE之间拼接最优模型。',
            '', '时序和Local摘要没有提高这轮整体AUC，不能据此断言原始Local D2无信息：输入摘要、训练目标及聚合已经不同于原real-only方法。监督结果与旧0.874472的差别同时包含fake训练信息、判别目标与摘要/聚合改变，不能称为MoE单因素增益。',
            '', '本轮停止增加专家数、路由损失或按测试调整训练数据。原论文主线不自动替换；若后续采用监督版本，必须重新明确论文的训练数据条件、方法贡献及困难域低误报边界。','']
        decision['external_run']=True
        decision['external_verification']=file_digest(external/'verification.json')
        decision['status']='requires_further_confirmation' if qualified else 'completed_no_stable_moe_gain'
    paper_json(out/'decision.json',decision)
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    paper_json(out/'manifest.json',dict(status=decision['status'],training_protocol=TRAINING_PROTOCOL,configuration=c,tests=tests,
        files={p.name:file_digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
