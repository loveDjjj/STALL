"""从已验收产物导出后续归因报告，保留正负结果及原始精度。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from statistical_experts.controls import prepare, check_files


LABELS={
 'official_final':'官方空间＋官方时序',
 'official_s_pooled_t':'官方空间＋同池总体时序',
 'official_s_offline_t_b':'官方空间＋离线时序，B整体CDF',
 'official_s_online_t_b':'官方空间＋在线时序，B整体CDF',
 'official_s_random_t':'官方空间＋随机时序，B整体CDF',
 'official_s_offline_t_a':'官方空间＋离线时序，A同专家全库CDF',
 'pooled_s_offline_t_b':'同池空间＋离线时序，B',
 'pooled_s_offline_t_a':'同池空间＋离线时序，A',
 'offline_t_b':'离线时序单分支，B',
 'offline_t_a':'离线时序单分支，A',
}


def score_table(frame, variants):
    lines=['| 配置 | AUC | AP-real | AP-fake | Recall@1% FPR |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name in variants:
        r=frame.loc[name]
        lines.append(f'| {LABELS.get(name,name)} | {r.auc:.6f} | {r.real_positive_ap:.6f} | {r.fake_positive_ap:.6f} | {100*r.fake_tpr_at_1pct_real_fpr:.2f}% |')
    return '\n'.join(lines)


def interval_table(frame):
    lines=['| 配对对比 | ΔAUC [95% CI] | ΔAP-real [95% CI] |',
           '| --- | ---: | ---: |']
    for name,p in frame[frame.dataset.eq('Average')].groupby('contrast',sort=False):
        def field(metric):
            r=p[p.metric.eq(metric)].iloc[0]
            return f'{r.delta:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]'
        lines.append(f'| {name} | {field("auc")} | {field("ap_real")} |')
    return '\n'.join(lines)


def report(root):
    root=Path(root);spec=prepare(root);out=root/spec['config']['run_directory']
    tests=ET.parse(out/'tests.xml').getroot()
    suites=list(tests.iter('testsuite'))
    test_count=sum(int(s.attrib.get('tests',0)) for s in suites)
    if not test_count or any(int(s.attrib.get('errors',0))+int(s.attrib.get('failures',0)) for s in suites):
        raise ValueError('测试证据缺失或存在失败')
    for name in ['verification.json','independent_audit.json','moments/verification.json']:
        if json.loads((out/name).read_text())['status']!='verified':raise ValueError('验收未齐备')
    for name in ['evaluation_manifest.json','analysis_manifest.json']:
        check_files(out,name);check_files(out/'moments',name)
    summary=pd.read_csv(out/'macro_metrics.csv').set_index('variant')
    intervals=pd.read_csv(out/'confidence_intervals.csv')
    moments=pd.read_csv(out/'moments/macro_metrics.csv').set_index('variant')
    mi=pd.read_csv(out/'moments/confidence_intervals.csv')
    primary=list(LABELS)[:6]
    lines=['# 官方空间、时序专家与参考校准：后续受控结果','',
        '本轮完成两组计划实验，并在时序收益通过配对检查后继续完成均值/协方差四格归因。原论文主线未替换。',
        '', '**结论：保留官方空间后，在线时序专家相对官方的Average AUC/AP差值区间均为正；离线A校准组合也超过官方，但A相对B存在AP-real损失。四格显示时序收益几乎全部跟随专家协方差，均值改变仅有极小影响。**',
        '', '## 1. 方法与数据边界','',
        '- 原版单窗Global；无Local D2、无Feature-change K3；空间max、时序min、两分支等权。',
        '- 沿用2200条VATEX fit、另2000条VATEX CDF；未新增目标real，未改变路由、近邻或Gaussian拟合。',
        '- 8/16帧分别校准，参考源视频身份相同，不把两种观察算作4000个独立源。',
        '- 15569个评价片段身份，23个生成器单元；Average=域内生成器等权后，三个数据集等权。',
        '- 所有Recall@FPR来自评价ROC，不是独立阈值迁移保证；本研究没有独立部署阈值集。',
        '- 参数不以fake调优；此次分支组合是看过首轮结果后的开发选择，区间不校正全部历史模型搜索。',
        '', '## 2. 保留官方空间的结果','',score_table(summary,primary),'',
        '相同官方空间固定后，用同池总体时序控制参考来源；在线随机512用于控制邻居预算。不能把不同来源的官方空间组合称为全部统计模型同池的实验。',
        '', '## 3. A与B改变了什么','',
        '**B：** 每条CDF视频选择自己的专家，得到原始时序分数，合并为一张整体算法CDF。',
        '', '**A：** 每个固定专家分别给全部2000条CDF视频评分；查询选择专家m后，使用该专家的全库参考。',
        '', '$$F_m(q)=N^{-1}\sum_j\mathbf 1[q_m(V_j^r)\le q],\qquad T_A(V)=F_{m(V)}(q_{m(V)}(V)).$$',
        '', 'A没有删除参考、没有只取同簇人口、没有新增一层CDF。A/B的Gaussian、路由与查询raw完全相同。两者都是合法校准定义；A带来的变化不能宣传成新增原始取证信息。',
        '',score_table(summary,list(LABELS)[6:]),'',
        '在官方空间下，A相对B的AUC差值区间跨零，AP-real差值区间为负。A相对官方的AUC/AP区间为正，但不证明A支配B。A提高了当前融合的AP-fake与Recall@1%点估计，说明不同排序区域之间存在取舍。',
        '', '## 4. 三域结果','',
        '| 配置 | ComGenVid AUC/AP | GenVideo AUC/AP | VideoFeedback AUC/AP |',
        '| --- | ---: | ---: | ---: |']
    ds=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset'])
    for name in primary:
        values=[]
        for d in ['comgenvid','genvideo','videofeedback']:
            r=ds.loc[(name,d)];values.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+LABELS[name]+' | '+' | '.join(values)+' |')
    lines += ['', 'A/B的域差异需保留：A在VideoFeedback提高AUC，而GenVideo的AP-real下降；不按域选择A或B。所有23单元及分母见[generator_metrics.csv](generator_metrics.csv)。',
        '', '## 5. 配对区间','',interval_table(intervals),'',
        '9个对比，每个1000次源组配对Poisson bootstrap。共享真实源在不同生成器中同步加权；95%区间条件于固定fit/CDF与路由seed，未作多重比较校正，不覆盖参考库重抽样不确定性。',
        '', '## 6. 时序均值/协方差四格','',
        '继续条件已通过：官方空间固定时，在线时序对同池总体时序的AUC和AP-real差值区间均为正。四格只交换既有离线时序模型的均值与Cholesky；协方差仍是原有0.5簇统计＋0.5总体统计及固定ridge。',
        '', '四组各自重建B式整体CDF。没有依据A/B结果替四格挑选更有利的校准方式。空间固定官方，每组另外报告时序raw-only与校准后单分支。',
        '', '| 均值 | 协方差 | raw-only AUC/AP | 时序校准后 AUC/AP | Final AUC/AP |',
        '| --- | --- | ---: | ---: | ---: |']
    for name,mu,cov in [('pooled','总体','总体'),('expert_mean','专家','总体'),
                        ('expert_covariance','总体','专家'),('expert_both','专家','专家')]:
        values=[]
        for branch in ['raw','temporal','final']:
            r=moments.loc[name+'_'+branch];values.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append(f'| {mu} | {cov} | '+' | '.join(values)+' |')
    lines += ['',interval_table(mi),'',
        '仅改均值的Final约0.832360/0.837337，几乎停留在总体0.832329/0.837316；仅改协方差约0.835077/0.843424，几乎恢复完整专家0.835075/0.843440。raw-only与校准后单时序也呈相同结构，故不是仅靠末端融合制造的现象。结论限于当前Global T1、参考池与收缩协议，不推广成所有模型的均值都无效。',
        '',
        'raw-only的原始+inf保留在raw_temporal_score。2条全静态评价片段使用原版时序边界；为兼容排序指标库，raw-only final_score采用保序且保同分的有限秩。该秩不用于融合或参考拟合。四格的CDF均由相应模型重新评分，未拼接不同模型旧参考。',
        '', '## 7. 成本、验收与复用','']
    for length in (8,16):
        cdf=json.loads((out/f'cdf_scores_{length}.json').read_text())
        m=json.loads((out/f'moments/raw_{length}.json').read_text())
        lines.append(f'- {length}帧：4专家×2000参考评分阶段{cdf["elapsed_seconds"]:.2f}s；四格全部{m["rows"]}条参考/评价查询{m["elapsed_seconds"]:.2f}s。两种长度分别在GPU0/GPU1运行；不含DINO提取，因为直接复用已有Global缓存。')
    lines += [f'- 完整测试：{test_count}项通过；详见tests.xml。',
        '- A的新增参考矩阵约125KiB（两个长度合计float64分数，不含文件元数据）；不保存每视频协方差或Patch。',
        '- 原路由对应参考raw、B百分位、四格两端点raw与Final均逐位恢复首轮；独立直接计数复核15569条A查询及10种融合/输出。',
        '- 四格首次数值检查发现显式批复制改变总体计算末位，最大约6.82e-13；恢复首轮广播形状后重算，未放宽容差。失败尝试未产生科学结果表。',
        '- 运行入口、数值边界及恢复规则见[运行手册](../../../docs/GLOBAL_EXPERTS_RUNBOOK_zh.md)。原论文、原视频和原参考包没有被覆盖。',
        '', '## 8. 后续取舍','',
        '1. 本轮支持把官方空间＋内容相关时序作为新研究候选，保留原论文主线。在线B与离线A都值得保留为比较行，不根据每个域挑更高的版本。',
        '2. 不把A校准直接宣布为B的替代：官方空间下A/B的AUC差值区间跨零，AP-real有下降；低误报点估计改善尚无专门区间。',
        '3. 不增加内容均值中心。下一步若进一步建模，应围绕时序方向协方差与参考支持，而非假设更多均值能带来收益。本轮没有运行新的子空间、邻居数或融合权重搜索。',
        '4. 收益幅度约为相对官方0.36–0.44个AUC百分点，尚不足以称大幅突破；若准备主方法更换，应先在新确认身份上冻结验证。',
        '', '## 9. 产物入口','',
        '- [逐视频分数](video_scores.csv.gz)、[三域](dataset_metrics.csv)、[23单元](generator_metrics.csv)、[配对区间](confidence_intervals.csv)。',
        '- [A/B分数移动诊断](score_changes.csv)、[专家参考分布](cdf_diagnostics.csv)。',
        '- [四格逐视频](moments/video_scores.csv.gz)、[四格23单元](moments/generator_metrics.csv)、[四格区间](moments/confidence_intervals.csv)。',
        '- identity.json绑定输入/模型/配置/核心源码；source_snapshot与moments/source_snapshot.py提供本次实际统计代码。',
        '- verification.json、independent_audit.json、moments/verification.json分别提供指标与逐条公式核验。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    extra=[Path('src/statistical_experts')/name for name in ('controls_report.py','controls_audit.py','run.py')]
    extra += [Path('tests/test_expert_calibration.py'),Path('tests/test_statistical_experts.py')]
    for relative in extra:
        destination=out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,destination)
    paper_json(out/'status.json',dict(status='completed',updated_utc=datetime.now(timezone.utc).isoformat(),
        studies=['official_spatial_temporal_hybrids','offline_same_expert_cdf','conditional_temporal_mean_covariance'],
        contrasts=14,evaluation_clip_ids=15569,generator_cells=23))
    files={}
    for p in out.rglob('*'):
        if p.is_file() and p!=out/'manifest.json' and 'moments_numerical_check' not in p.parts:
            files[str(p.relative_to(out))]=file_digest(p)
    def git(*args):return subprocess.check_output(['git',*args],cwd=root)
    paper_json(out/'manifest.json',dict(status='completed',source_identity=spec,
        git_head=git('rev-parse','HEAD').decode().strip(),git_branch=git('branch','--show-current').decode().strip(),
        tracked_diff_sha256=hashlib.sha256(git('diff','--binary')).hexdigest(),
        status_sha256=hashlib.sha256(git('status','--porcelain')).hexdigest(),
        python=sys.version,tests_passed=test_count,
        report_code_sha256=file_digest(Path(__file__)),files=files,
        exclusions='moments_numerical_check仅保留失败的数值检查身份，不属于科学结果',
        interpretation='固定参考的开发探索，不替换原论文主线；不宣称所有指标或所有域改善'))
    print(out/'RESULTS_zh.md',flush=True)
