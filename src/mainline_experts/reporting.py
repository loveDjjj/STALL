"""四格原主线专家结果：数据条件、成本和继续判定分别报告。"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from mainline_experts.run import configuration


def report(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];v=json.loads((out/'verification.json').read_text())
    if v['status']!='verified':raise ValueError('完整验收尚未完成')
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) for s in suites):raise ValueError('测试证据不完整')
    tests=sum(int(s.attrib['tests']) for s in suites);m=pd.read_csv(out/'macro_metrics.csv').set_index('variant');d=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);ci=pd.read_csv(out/'confidence_intervals.csv')
    labels={'R0':'当前主线','R1':'仅GT协方差专家','R2':'仅Local D2协方差专家','R3':'GT＋Local专家'}
    lines=['# 原主线上的目标协方差专家：开发全量四格','',
        '本轮基线是目标Global＋目标Local D2＋FC K3，最新23单元0.874472/0.877075；不是native单窗Global专家。空间分支不变，两个专家、0.5目标总体协方差收缩、共享原均值，不新增最终评分分支。',
        '', '## 方法与数据','',
        '- 每域沿原200拟合片段（ComGenVid132源），Uniform K3、每视频256 D2、片段等权。专家继承样本原权重，不把单个落入某簇的位置赋整视频权重。',
        '- 内容描述来自该窗口Global；GT/Local共用真实拟合集学习的两中心硬路由。拟合、CDF和评价身份分离，不用fake拟合或选权重。',
        '- GT专家CDF来自同2000 VATEX的Uniform首窗；Local专家CDF来自FC K3及原ranked-linspace effective-K规则；8帧另有相应参考。保留主线原有规则，不加CDF层级。',
        '- 一次每视频的原batch8/union前向服务四格；CDF相同物理窗口跨域去重后分别评分。没有重新使用旧native单窗或假定Uniform Patch全部命中FC。',
        '', '## 主比较','', '| 配置 | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Average AUC/AP |','| --- | ---: | ---: | ---: | ---: |']
    for key,label in labels.items():
        vals=[]
        for domain in ('comgenvid','videofeedback','genvideo'):
            q=d.loc[(key,domain)];vals.append(f'{q.auc:.6f}/{q.real_positive_ap:.6f}')
        q=m.loc[key];vals.append(f'{q.auc:.6f}/{q.real_positive_ap:.6f}');lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines += ['', 'Average先域内生成器等权，再三个域等权；不与旧20单元0.881462相减。',
        '', '## 分支与操作点','', '| 输出 | AUC | AP-real | AP-fake | Recall@1% FPR |','| --- | ---: | ---: | ---: | ---: |']
    for key in ['GT0','GT1','global0','global1','local0_raw','local1_raw','local0','local1','R0','R1','R2','R3']:
        q=m.loc[key];lines.append(f'| {key} | {q.auc:.6f} | {q.real_positive_ap:.6f} | {q.fake_positive_ap:.6f} | {100*q.fake_tpr_at_1pct_real_fpr:.2f}% |')
    lines += ['', '## 源级配对区间','', '| 对比 | ΔAUC [95% CI] | ΔAP-real [95% CI] |','| --- | ---: | ---: |']
    for name,q in ci[ci.dataset.eq('Average')].groupby('contrast',sort=False):
        vals=[]
        for metric in ('auc','ap_real'):
            r=q[q.metric.eq(metric)].iloc[0];vals.append(f'{r.delta:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '8组对比，每组1000次固定源组Poisson重抽样。参考池和当前参数固定，区间未多重校正，不包含换真实拟合集的波动。Recall来自评价ROC，不是独立阈值部署保证。',
        '', '## 真实留出与支持','', '| 域 | 分支 | 总体NLL | 专家NLL |','| --- | --- | ---: | ---: |']
    for domain in c['datasets']:
        h=pd.read_csv(out/f'fit_{domain}/holdout.csv').groupby('branch')[['pooled_nll','expert_nll']].mean()
        for b,q in h.iterrows():lines.append(f'| {domain} | {b} | {q.pooled_nll:.6f} | {q.expert_nll:.6f} |')
    lines += ['', '总体先验、中心和专家都在源级留出训练部分重拟合。NLL仅诊断；尤其GT的VideoFeedback留出变差，不因某个Final数字较高而省略。每簇片段/窗口/独立源/有效源权重见all_fit_support.csv，少源簇不等于大量独立Patch。',
        '', '## 验收与后续','',
        f'- {tests}项tests通过；六个原始拟合视频的Global/D2抽样逐元素一致，全部15569视频M1/完全回退CDF融合公式恢复原结果。',
        f'- 全量R0原始分数最大误差{v["real_forward_R0_max_error"]:.3g}，重算CDF后Final最大误差{v["R0_final_recomputed_max_error"]:.3g}；每视频四格加法恒等式通过。',
        f'- 独立检查{v["route_checks"]}个窗口路由、{v["expert_percentile_checks"]}个专家百分位与{v["original_video_expert_probes"]}个原图专家探针。',
        '- 只有独立增益明确的分支才进入拟合内容归属打乱控制及冻结外部验证；开发结果未确认前，不替换论文主线。若无清楚收益，停止本次主线专家假设，不继续扫描。',
        '- [23单元](generator_metrics.csv)、[逐视频](video_scores.csv.gz)、[全部区间](confidence_intervals.csv)、[R0回归](R0_regression.csv)、[拟合支持](all_fit_support.csv)。','']
    completions=[json.loads(p.read_text()) for p in sorted((out/'dense').glob('completed_*.json'))]
    plans=json.loads((out/'plans.json').read_text())['jobs']
    dense_frames=sum(len({i for w in j['windows'] for i in w}) for j in plans)
    uniform_frames=sum(len(j['uniform']) for j in plans)
    lines += ['## 计算成本','',
        f'- 共{len(plans)}个物理任务；FC窗口合并后{dense_frames}个前向帧，另有{uniform_frames}个Uniform参考前向帧；不含尾批填充。',
        '- 双卡同一轮前向同时服务四格。以下为研究评分任务耗时，包含原始基线逐窗核验和全部专家对照，不能当作候选模型的部署延迟。',
        *[f'- GPU rank {x["rank"]}：{x["seconds"]/60:.2f}分钟；该轮复用已通过的pilot检查点。' for x in completions], '']
    primary=ci[ci.dataset.eq('Average') & ci.contrast.isin(['R1_vs_R0','R2_vs_R0','R3_vs_R0'])]
    no_positive=not (primary.ci95_low>0).any()
    decision=dict(status='completed_no_go' if no_positive else 'requires_followup_review',
        criterion='预定主要对比至少一个指标有可靠正增量；当前三候选六个Average区间均无正下界。' if no_positive else '需依原方案审查独立增益与损害，再确定打乱及外部验证。',
        baseline_unchanged=True,shuffle_control='not_triggered' if no_positive else 'pending_review',
        external_confirmation='not_triggered' if no_positive else 'pending_review',
        evidence_sha256=file_digest(out/'confidence_intervals.csv'))
    paper_json(out/'decision.json',decision)
    if no_positive:
        lines += ['## 本轮决定：不采纳专家，保留原主线','',
            'R1、R2、R3相对R0的Average六个主要指标区间均没有正下界。R2的微小点估计增益不满足可靠新增收益门槛；不启动以通过该门槛为前提的拟合内容打乱或外部确认，不继续扫专家数、收缩或CDF。',
            '', 'R2在ComGenVid和GenVideo有不同程度正信号，但VideoFeedback的Final AUC/AP下降区间均在零以下。VideoFeedback的Local原始AUC已经下降0.013918，因而不能把失败全部归咎于CDF；GT原始校准分支AUC下降0.074321，双专家进一步恶化该域。',
            '', '这不是专家实现未生效：专家原始分数改变、矩阵独立核验和R0完全回归均通过。更细的内容协方差可能改变真实方向惩罚并造成域相关利弊；当前实验不能唯一确定是内容分组、有限源支持还是估计偏差导致，不把这种解释当因果证明。',
            '', '真实留出NLL与检测收益并不等价。少源/不均衡簇是限制，但没有做参考重抽样，不能断言样本不足就是唯一原因。外部效果本轮未测，也不据此声称所有专家方案均无效。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    for p in ['src/mainline_experts/evaluation.py','src/mainline_experts/audit.py','src/mainline_experts/reporting.py','src/mainline_experts/models.py','src/mainline_experts/run.py','src/mainline_experts/experiment.py','configs/mainline_experts.yaml']:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
    paper_json(out/'manifest.json',dict(status=decision['status'],tests_passed=tests,configuration=c,
        files={str(p.relative_to(out)):file_digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
