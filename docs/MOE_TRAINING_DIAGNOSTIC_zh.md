# MoE容量与预热训练诊断

状态：2026-09-10已完成48模型、开发/外部评价和独立验收。上一轮已完成结果和论文主线不变。本轮只用Global摘要，沿用length_matched_supervision_v2的源级split、真假帧数匹配、采样权重与三seed。

| 变体 | 结构 | 日程 |
| --- | --- | --- |
| wide_mlp | 隐藏宽256的单MLP | 30轮 |
| uniform | 两个128宽专家，固定平均 | 30轮 |
| joint | 相同两个128宽专家＋线性softmax gate | 从第1轮联合训练，共30轮 |
| warm | 与joint完全相同 | 前10轮gate固定为均匀；第11轮解冻，共30轮 |

四组不改变数据、损失超参数或推理概率定义；每fold/seed采样序列相同。warm不增加训练轮数，不重置专家优化器；gate首次解冻时建立自身AdamW状态。第10轮保存全部模型状态，验证warm的专家与独立uniform前10轮在相同seed下对齐。所有专家始终计算，不把此实验称为稀疏LLM MoE。

每个模型只按训练侧验证Average(AUC,AP-real)选择epoch。warm在前10轮也可成为最佳模型，必须如实说明若最终选中的是尚未启用路由的阶段；另保存final模型用于训练轨迹核验，不根据测试改用final。共4变体×4fold（含pooled）×3seed=48个模型。

训练每epoch记录：实际updates和抽样序列hash、总loss、视频平均gate质量/窗口熵、各专家/路由器裁剪前梯度范数、每专家独立验证NLL及按内容簇/真实来源/生成器的损失。两个内容簇只用该fold训练Global描述拟合，并按既有采样权重加权；不进入训练目标，不将其直接命名为语义专家。

对照和判定：joint−uniform定位同容量路由作用；warm−joint定位预热作用；warm−uniform检查预热后是否超过简单平均；各组对wide_mlp比较。上轮G窄MLP、64宽双头及MoE作为已有预算参照单列。不能把扩大容量共同带来的收益归给路由，不能把训练loss下降当跨域收益。

首先完成48模型并冻结选择，再统一评价开发23单元和既有外部20单元，不按域选配置。外部已被观察，不声称全新盲测；域留出不等于生成器家族留出。开发和外部都有逐seed、逐生成器、AUC/AP-real/AP-fake/ROC操作点及源级配对区间；不同模型的参考数据条件一致，不使用目标Gaussian/CDF。

回归与验收：输入/源码/模型hash、源级隔离及长度匹配、warm阶段gate零参数/零梯度、epoch10专家对齐、实际updates相同、独立NumPy预测与标准化复核、全部指标重算。若无稳定收益，就保留简单分类器，不再扫warmup轮数、学习率或专家数。

代码src/moe_training，配置configs/moe_training.yaml，产物results/runs/moe_training。只读复用特征，不重新提DINO，不改src/discriminative_moe中的已验收实现。

## 已完成结果

| 配置 | 开发Average AUC/AP-real | GenVidBench AUC/AP-real | ViF AUC/AP-real |
| --- | ---: | ---: | ---: |
| 旧2×64均匀头 | 0.967776/0.967489 | 0.958243/0.955983 | 0.748688/0.740938 |
| 旧2×64 MoE | 0.963824/0.964721 | 0.954556/0.956435 | 0.739100/0.745337 |
| 宽256 MLP | 0.965796/0.965423 | 0.961243/0.958530 | 0.740464/0.739481 |
| 2×128均匀头 | 0.966954/0.966638 | 0.964272/0.965552 | 0.740542/0.743994 |
| 2×128联合MoE | 0.967168/0.967329 | 0.955606/0.955777 | 0.737613/0.737993 |
| 2×128预热MoE | 0.966818/0.966453 | 0.964272/0.965552 | 0.740542/0.743994 |

joint对同容量uniform ΔAUC +0.000214，95%区间[-0.001225,+0.002077]；warm对joint -0.000350，区间[-0.001885,+0.000904]，均无可靠增益。joint相对旧窄MoE +0.003344 [0.001783,0.004983]，表明容量影响了开发表现，但未形成一致的外部优势。

pooled三个warm最佳epoch为6/7/4，与uniform相同，尚未启用gate；外部结果相同不是新的路由收益。88项tests通过；48模型的抽样序列、updates一致，12对warm/uniform第10轮专家逐参数一致；216个独立NumPy预测探针，最大误差1.72e-7。开发23和外部20单元指标全部重算通过。固定此结果，不继续扫描预热轮数、学习率或gate温度。

入口：`python -m src.moe_training.run prepare`；两卡分别`train --rank 0/1 --world-size 2`；完成后依次`evaluate`、`analyze`、`verify`、`report`。配置及恢复只接受相同输入/源码hash。完整[报告](../results/runs/moe_training/RESULTS_zh.md)、[逐生成器](../results/runs/moe_training/generator_metrics.csv)、[区间](../results/runs/moe_training/confidence_intervals.csv)、[epoch选择](../results/runs/moe_training/selected_epochs.csv)。
