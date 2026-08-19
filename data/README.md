# 数据清单

`manifests/` 是 Alpha STALL 的数据身份来源，不存放视频本身。当前实验有三个
开发数据集和一个外部数据集；它们不是不同方法的数据。

| 目录 | 用途 |
|---|---|
| `manifests/development/` | ComGenVid、GenVideo、VideoFeedback 的真实校准集和评测集 |
| `manifests/external/` | 外部 GenVidBench 评测的真实校准集和评测集 |

当前切分如下：

| 数据集 | 校准集 | 评测集 | 用途 |
|---|---:|---:|---|
| ComGenVid | 200 real | 900 real + 3,400 fake | 开发集 |
| GenVideo | 200 real | 7,984 real + 8,204 fake | 开发集 |
| VideoFeedback | 200 real | 500 real + 3,000 fake | 开发集 |
| GenVidBench Pair1 `ms`/`vript` | 199 real | 300 real + 300 fake | 冻结后的外部泛化 |

每个 manifest 至少包含 `video_path`、`subset` 和 `source_model`。视频路径相对
仓库根目录，例如 `datasets/genvideo/real/...`。新数据集使用
`scripts/build_manifest.py` 生成带采样帧索引的 manifest，再按校准集与评测集拆分。

当前默认方法使用 2 秒、16 帧窗口。`configs/benchmark.yaml` 的
`data.short_video_policy` 默认是 `exclude`：不足 16 个下采样帧的视频不满足固定 2 秒
方法的输入要求，因而会被一致地排除；运行 manifest 会逐数据集记录排除数量，避免评测
时静默删样本。需要将短视频视为协议错误时，可显式覆盖为 `error`。
