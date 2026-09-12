"""首轮Global统计专家结果，不将候选实验覆盖到既有论文主线。"""
import json
from pathlib import Path
import subprocess
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from statistical_experts.manifests import settings

LABELS={'official':'官方STALL（发布NPZ）','pooled':'同池总体模型','offline':'离线4专家','online':'在线相似512','random':'在线随机512'}


def report(root):
    root=Path(root);c=settings(root);run=root/c['run_directory'];ev=run/'evaluation'
    audit=json.loads((run/'verification.json').read_text())
    if audit['status']!='verified':raise ValueError('研究未完成验收')
    domains=pd.read_csv(ev/'dataset_metrics.csv');summary=pd.read_csv(ev/'summary.csv');ci=pd.read_csv(ev/'confidence_intervals.csv')
    cost=pd.read_csv(run/'runtime/summary.csv');hybrids=pd.read_csv(ev/'hybrid_macro_metrics.csv')
    lines=['# Global统计专家首轮结果','',
        '已完成：总体、离线4专家、在线相似512、在线随机512，以及官方发布参数锚点。仅使用原版单窗Global空间/时序，不含Local D2、Feature-change K3或新融合权重。','',
        '**结论：内容相关统计相对同池总体模型有正向增益；在线没有显著超过离线。当前更值得优先保留离线专家作为后续研究候选，而不是立即替换原论文主线。**','',
        '## 1. 完整检测结果','',
        '| 方法 | VideoFeedback AUC/AP | GenVideo AUC/AP | ComGenVid AUC/AP | Average AUC/AP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for method in ('official','pooled','offline','online','random'):
        values=[]
        for domain in ('videofeedback','genvideo','comgenvid'):
            r=domains[(domains.method==method)&(domains.branch=='final')&(domains.dataset==domain)].iloc[0]
            values.append(f'{r.auc:.3f}/{r.real_positive_ap:.3f}')
        r=summary[(summary.method==method)&(summary.branch=='final')].iloc[0]
        values.append(f'{r.auc:.3f}/{r.real_positive_ap:.3f}')
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    lines+=['','Average为三个数据集等权平均，AP指AP-real。全部23单元见[evaluation/generator_metrics.csv](evaluation/generator_metrics.csv)。','',
        '同池四组共用2200条VATEX fit、2000条独立VATEX CDF；没有使用目标域真实拟合。官方NPZ来自更大真实参考且建库协议不同，因此主要归因看同池控制，而非只看官方差值。','',
        '## 2. 配对差值','',
        '| 对比 | AUC差值及95% CI | AP差值及95% CI |','| --- | ---: | ---: |']
    names={'offline_vs_pooled':'离线−总体','online_vs_pooled':'在线−总体','online_vs_random':'在线−随机',
        'online_vs_offline':'在线−离线','online_vs_official':'在线−官方','offline_vs_official':'离线−官方'}
    for key,label in names.items():
        values=[]
        for metric in ('auc','ap_real'):
            r=ci[(ci.dataset=='Average')&(ci.contrast==key)&(ci.metric==metric)].iloc[0]
            crossing='；跨零' if r.ci95_low<=0<=r.ci95_high else ''
            values.append(f'{r.delta:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]{crossing}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','离线/在线相对总体均有正区间，在线相对随机也有正区间；在线与离线的两项差值都跨零。与官方NPZ相比，两种专家的AUC差异均未证明为正，AP有正向区间。不能写“全面超过原版”。',
        '区间是1000次源组配对Poisson bootstrap，条件于固定fit/CDF和随机seed，未多重比较校正；不包含更换参考库的不确定性。','',
        '## 3. 两条分支','',
        '| 方法 | 空间-only Average AUC/AP | 时序-only Average AUC/AP |','| --- | ---: | ---: |']
    for method in ('pooled','offline','online','random'):
        values=[]
        for branch in ('spatial','temporal'):
            r=summary[(summary.method==method)&(summary.branch==branch)].iloc[0];values.append(f'{r.auc:.3f}/{r.real_positive_ap:.3f}')
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    lines+=['','保持另一分支为同池总体，进一步组合：','',
        '| 变体 | Average AUC/AP |','| --- | ---: |']
    for r in hybrids.itertuples():lines.append(f'| {r.variant} | {r.auc:.3f}/{r.real_positive_ap:.3f} |')
    lines+=['','这里不是又加了新模型：直接组合已校准的两个分支分数，顶层仍0.5/0.5。在线时序替换的增益大于在线空间替换，但不能据点估计声称只有时序有用。离线仅时序替换的AP区间为正，AUC区间跨零。','',
        '## 4. 实际计算成本','',
        '| 方法 | 8帧统计毫秒/查询 | 16帧统计毫秒/查询 |','| --- | ---: | ---: |']
    for method in ('pooled','offline','online','random'):
        values=[f'{1000*cost[(cost.method==method)&(cost.length==length)].mean_seconds_per_query.iloc[0]:.3f}' for length in (8,16)]
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    cache_elapsed=max(json.loads(p.read_text())['elapsed'] for p in (run/'cache').glob('progress_rank_*.json'))
    score_elapsed=sum(max(json.loads(p.read_text())['elapsed'] for p in (run/f'raw_{length}').glob('completed_rank_*.json')) for length in (16,8))
    lines+=['',
        '这是Global特征已就绪后的实际算法＋CDF成本，batch4摊销口径，不含DINO、解码和磁盘读取，不能称单请求端到端延迟。60个固定评价片段、四算法、两遍，共480个计时查询；输出与正式分数一致。',
        f'双卡各两个进程：Global缓存核心阶段最长worker约{cache_elapsed/60:.1f}分钟；四算法共享统计的长窗/短窗阶段合计约{score_elapsed/60:.1f}分钟，另有模型初始化、CPU分析与实现时间。缓存约1.4GiB，未新增Patch库。',
        '统计阶段的GPU利用率实际观察到约98–100%；解码、预处理、Global提取及异步写盘采用有界流水线。未记录连续功耗曲线，不将瞬时利用率当全程平均。','',
        '## 5. 真实统计诊断','',
        '在2200 fit内部独立划分1760训练/440留出，重新拟合路由和模型。专家时序NLL低于总体，且优于固定无关专家；空间NLL反而比总体更高。因此不能宣称多专家全面更准确拟合真实分布，检测排序改善与真实预测密度改善不是同一件事。',
        'NLL在共同1024维正定Gaussian近似下包含logdet，低者更好；该logdet仅用于诊断，检测仍是约定的高斯能量＋整流程CDF。详见[real_diagnostic/summary.csv](real_diagnostic/summary.csv)。','',
        '## 6. 协议与验收','',
        '- 2200 fit来自旧source-fit200＋旧threshold2000；CDF是另外2000。旧threshold不再用作此研究独立阈值，不声称部署FPR保证。',
        '- 8/16帧各有对应模型和CDF，相同原视频身份划分；原生窗口、224预处理、空间拟合帧索引均固定。',
        '- 在线/随机每查询512条fit邻居，协方差统一0.5总体收缩、1e-5 ridge、完整1024维、float64统计。路由只用内容，随机控制用固定匿名源键。',
        '- 四种算法都采用B式整流程CDF：独立CDF视频各自运行检索/路由/拟合/评分算法，之后建立两分支分布。不是旧方案每个query重评分整个参考库。',
        '- 固定参考下已核验23969窗口缓存、19569条统计查询和同数量随机邻居规则，参考/评价未发现完全相同Global窗口。不是完整语义去重。',
        '- 200条真实窗口做像素级解码对照；所有评价缓存另匹配官方GS/GT摘要和Final锚点。没有把标量一致冒充全量逐帧特征与历史缓存逐元素一致。',
        '- 原论文主线未替换，新分支为research/global-statistical-experts；原工作区已保存源码及tracked diff快照，既有未提交改动保留。','',
        '## 7. 后续建议','',
        '当前通过了“内容相关统计值得继续”的首轮门槛，尚未证明在线优于离线。优先以离线4专家继续机制验证，在线保留为灵活参考选择的对照。下一项最有价值的是固定当前协议拆分专家均值与协方差，再判断是否增加参考规模；不直接增加到100专家或混入Local/FC。',
        '当前0.832/0.833的Global结果与使用目标real、Local D2、多窗口的原论文0.874不是同一条件，不能直接相减作为新方法成败。']
    (run/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
    code={str(p.relative_to(root)):file_digest(p) for p in (root/'src/statistical_experts').glob('*.py')}
    paper_json(run/'manifest.json',dict(status='completed',branch=branch,protocol=c,
        verification=audit,code=code,source_before=file_digest(root/'results/runs/global_experts_setup/paper_source_before.tar.gz'),
        files={str(p.relative_to(run)):file_digest(p) for p in run.rglob('*') if p.is_file() and p!=run/'manifest.json' and 'checkpoints' not in p.parts and 'before_metadata_fix' not in str(p)}))


if __name__=='__main__':report(Path.cwd())
