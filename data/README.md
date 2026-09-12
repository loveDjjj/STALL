# 数据清单

本目录保存身份、角色与迁移记录；视频本身在datasets，特征在cache。

- manifests/active：拟合、CDF、阈值及旧20单元评价入口，保留既有身份。
- manifests/short_video：完整23单元需要的8帧参考和评价扩展。
- 完整论文评价清单：results/paper_complete/evaluation.csv及pairs.csv。
- catalog：来源、保留/删除收据和路径映射。
- native_stall、global_experts等清单保留原实验身份，不自动加入当前评价。

CSV/JSON等冻结输入按原字节版本化；.gitattributes禁用数据换行转换，避免改变hash和抽样身份。不要运行代码格式化工具重写数据清单。

完整角色和缓存说明见[DATA_zh.md](../docs/DATA_zh.md)，短视频与当前范围见[协议说明](../docs/MANUSCRIPT_PROTOCOL_NOTES_zh.md)。
