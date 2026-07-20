# ComGenVid temporal derivative order 对照

本实验只在 ComGenVid 上补充 D=3/D=4 高阶同网格 patch temporal derivative 对照，用于回应“二阶局部时序证据是否随意选择”的审稿风险。
实验不铺开三数据集；它沿用当前 ComGenVid patch 主线设置：2 秒 compact cache、真实视频校准、patch region=3、bottom-k=0.20、pairwise balanced AUC/AP。

## 结果

| order | patch mode | 平均 AUC | 平均 AP | ΔAUC vs D=2 | ΔAP vs D=2 | 角色 |
|---:|---|---:|---:|---:|---:|---|
| 2 | `same_grid_second_order` | 0.9273 | 0.9309 | +0.0000 | +0.0000 | current Alpha-STALLED patch temporal branch |
| 3 | `same_grid_third_order` | 0.8853 | 0.8936 | -0.0420 | -0.0373 | higher-order derivative control |
| 4 | `same_grid_fourth_order` | 0.8866 | 0.8949 | -0.0407 | -0.0361 | higher-order derivative control |

## 结论

- 当前主线 D=2 的平均 AUC/AP 为 0.9273 / 0.9309。
- D=3 和 D=4 均低于 D=2；其中 D=4 为 0.8866 / 0.8949，相对 D=2 的 ΔAUC=-0.0407。
- 结果支持保留二阶局部时序证据作为 Alpha-STALLED patch 分支主线：三阶/四阶差分会强调更高频的局部变化，但在当前 16 帧、region=3、bottom-k=0.20 设定下没有带来收益。
- 该实验是代表性机制对照，不应写成三数据集鲁棒性结论；若审稿人要求完整 temporal derivative sweep，再扩展到 VideoFeedback/GenVideo。

## 重建命令

```bash
conda run --no-capture-output -n stall python src/create_patch_params.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --output precomputed/patch_params_comgenvid_real_same_grid_third_order_region3_bottomk0p20_v2.npz \
  --duration 2 --compact --real-only --max-patches-for-fit 300000 \
  --aggregation bottomk_mean --bottomk-ratio 0.20 \
  --patch-temp-mode same_grid_third_order --patch-region-size 3

conda run --no-capture-output -n stall python src/create_patch_params.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --output precomputed/patch_params_comgenvid_real_same_grid_fourth_order_region3_bottomk0p20_v2.npz \
  --duration 2 --compact --real-only --max-patches-for-fit 300000 \
  --aggregation bottomk_mean --bottomk-ratio 0.20 \
  --patch-temp-mode same_grid_fourth_order --patch-region-size 3
```

机器可读文件：

- `comgenvid_temporal_derivative_order_summary.csv`
- `comgenvid_patch_third_order.csv` / `comgenvid_patch_third_order_metrics.csv`
- `comgenvid_patch_fourth_order.csv` / `comgenvid_patch_fourth_order_metrics.csv`
- `comgenvid_temporal_derivative_order.svg` / `.png`
