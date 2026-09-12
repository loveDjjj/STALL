# 论文材料

当前稿位于[ieee_alpha_stalled](ieee_alpha_stalled/README.md)，采用中文IEEE双栏；流程图按作者要求暂留空。

数值来自当前23单元results/paper_complete和已完成的局部方向、新真实池验证。主表发表值与本次计算分组；消融固定配对身份和信息预算。旧20单元、监督训练与原生方法结果不能混为同一协议。

~~~bash
python3 paper/ieee_alpha_stalled/export_tables.py --check
bash paper/ieee_alpha_stalled/build.sh
~~~

当前仅维护三张自动导出表。旧表及旧方法图已从工作树移除，可从Git恢复点efede85查看。源码、生成表格和最终PDF纳入Git；编译中间文件忽略。results仅留本地，--check需本地CSV；新检出可直接编译已提交的TeX表格。
