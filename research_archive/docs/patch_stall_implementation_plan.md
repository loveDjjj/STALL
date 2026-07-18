# Patch-STALL / STE-STALL 实现计划

本文档用于指导在当前 `STALL` 代码库中逐步实现 patch-level 改进版。目标是在**不破坏原版可复现性**的前提下，新增一套 patch-level 空间/时序分支，并支持与原版 global STALL 融合。

---

## 0. 总体原则

实现时遵循以下原则：

1. 原版 `src/stall.py`、`src/eval.py`、`src/create_params.py` 尽量不做破坏性修改。
2. 优先新增文件，而不是重写原版逻辑。
3. 新增 patch 分支时，尽可能复用原版：
   - DINOv3 加载
   - whitening / Gaussian log-likelihood
   - percentile score
   - metrics 计算
4. 所有新功能先做最小可用版，再逐步加复杂特性。
5. 分数方向必须保持一致：
   - **分数越高，越像真实视频**

---

## 1. 新增文件

第一步先新增以下文件，不直接改坏原版：

```text
src/stall_patch.py
src/eval_patch.py
src/create_patch_params.py
src/dataset_utils_patch.py
src/patch_matching.py
tools/inspect_dinov3_tokens.py
```

文件职责建议如下：

- `src/stall_patch.py`
  - `PatchSTALL` 主类
  - patch/global embedding 抽取
  - patch spatial / patch temporal score 计算
- `src/eval_patch.py`
  - patch 版评估入口
  - 支持 `csv + patch cache + params`
- `src/create_patch_params.py`
  - 从真实视频或 patch cache 中重算 patch-level calibration params
- `src/dataset_utils_patch.py`
  - patch cache 读写
  - 参考原版 `dataset_utils.py`
- `src/patch_matching.py`
  - motion-aligned patch matching
- `tools/inspect_dinov3_tokens.py`
  - 验证 DINOv3 patch token 输出结构

---

## 2. 分阶段实现顺序

不要一步到位。推荐按下面顺序推进：

### 阶段 A：验证 DINOv3 patch token 输出

目标：

- 找到 DINOv3 patch token 字段名
- 确认 global embedding 与 patch token 可以同时拿到

实现：

- 编写 `tools/inspect_dinov3_tokens.py`
- 复用原版：
  - `load_dinov3_model()`
  - `create_dinov3_transform()`
- 读取一帧视频图像
- 尝试：
  - `model.forward_features(x)`
  - 如果失败，查看模型返回结构
- 打印：
  - 所有 key
  - 所有 shape

预期：

- `224x224` 输入下，ViT-L/16 patch token 数约为 `14x14=196`
- 形状类似：
  - patch tokens: `[B, 196, D]`
  - global embedding: `[B, D]`

这一阶段如果没确认清楚，后面先不要做。

---

### 阶段 B：实现 patch/global embedding 提取

目标：

- 能对视频抽出：
  - global embedding
  - patch token embedding

实现文件：

- `src/stall_patch.py`

核心函数：

```python
_embed_flat_frames_with_patches(flat_frames, batch_size=32)
```

返回：

```python
global_embs: np.ndarray   # [total_frames, D]
patch_embs: np.ndarray    # [total_frames, P, D]
grid_size: tuple[int, int]
```

再实现：

```python
frames_to_global_patch_embeddings(video_arrays, batch_size=32)
```

返回每个视频：

```python
{
    "global": np.ndarray,   # [T, D]
    "patch": np.ndarray,    # [T, P, D]
    "grid_size": [Gh, Gw],
}
```

注意：

- 第一版固定输入 `224x224`
- 暂不考虑 native resolution

---

### 阶段 C：实现 patch cache

目标：

- patch embedding 可缓存到磁盘，避免反复抽特征

实现文件：

- `src/dataset_utils_patch.py`

参考原版：

- `prefill_emb_cache()`
- `load_csv_with_emb_cache()`

建议 patch cache 格式：

```python
{
    "global": torch.Tensor[T, D],
    "patch": torch.Tensor[T, P, D],
    "grid_size": [Gh, Gw],
    "frame_indices": [...],
}
```

建议路径：

```text
cache/patch_embeddings/<dataset>/<subset>/<source_model>/<video_stem>_2s.pt
```

建议支持：

- `--compact`
- `--debug N`

优先先跑通一个小样本 debug 版本。

---

### 阶段 D：实现 patch spatial params

目标：

- 用真实视频 patch token 拟合真实局部空间分布

实现文件：

- `src/create_patch_params.py`

第一版只做：

- `patch spatial`

核心流程：

1. 读取真实视频 patch embeddings
2. flatten 成：

```python
flat_patch = patch_embs.reshape(-1, D)
```

3. 从中随机采样不超过 `--max-patches-for-fit`
4. 拟合：
   - `mu_patch_spat`
   - `W_patch_spat`
5. 对每个 calibration real 视频计算：

```python
patch_spat_ll: [N, T, P]
```

6. 聚合成每视频一个分数：

- `mean`
- `min`
- `bottomk_mean`

默认建议：

- `bottomk_mean`
- `bottomk_ratio=0.05`

输出至少包含：

```python
mu_patch_spat
W_patch_spat
calib_patch_spat_scores
patch_grid_size
duration
aggregation_config
```

注意：

- fake 视频绝对不能参与拟合
- 第一版先允许使用 benchmark real 的 calibration split 做工程原型

---

### 阶段 E：实现 patch spatial only 评估

目标：

- 跑通 patch-level 空间分支，得到可比较的 AUC/AP

实现文件：

- `src/eval_patch.py`

第一版只支持：

- 读取 patch cache
- 加载 patch spatial params
- 计算：
  - `patch_spat_percentile`
  - `patch_final_score`
- 输出结果 CSV
- 复用原版 `metrics.py`

建议先做：

```text
fusion = patch_only
disable_patch_temp = True
```

输出 CSV 至少包含：

```text
subset
source_model
filename
patch_spat_percentile
patch_final_score
final_score
```

---

### 阶段 F：实现 same-grid patch temporal

目标：

- 在不做 patch matching 的前提下，先得到局部时间分支

实现位置：

- `src/stall_patch.py`
- `src/create_patch_params.py`

做法：

输入：

```python
patch_embs: [N, T, P, D]
```

计算：

```python
delta = patch_embs[:, 1:, :, :] - patch_embs[:, :-1, :, :]
delta = l2_normalize(delta)
```

得到：

```python
delta: [N, T-1, P, D]
```

再拟合：

```python
mu_patch_temp
W_patch_temp
calib_patch_temp_scores
```

聚合也默认用：

- `bottomk_mean`

这一步实现后，可以支持：

- `patch spatial + same_grid temporal`

---

### 阶段 G：实现 global + patch fusion

目标：

- 把原版 global STALL 与 patch-level STALL 融合

做法：

```python
global_final = 0.5 * (global_spat_pct + global_temp_pct)
patch_final = 0.5 * (patch_spat_pct + patch_temp_pct)
final_score = 0.5 * global_final + 0.5 * patch_final
```

CLI 建议支持：

```bash
--fusion global_only
--fusion patch_only
--fusion avg
```

同时保留：

- `global_final_score`
- `patch_final_score`
- `final_score`

---

### 阶段 H：实现 motion-aligned patch temporal

目标：

- 在局部窗口内做 patch matching，而不是 rigid same-grid 对齐

实现文件：

- `src/patch_matching.py`

核心函数：

```python
match_patches_local_window(
    patch_t,
    patch_t1,
    grid_size,
    radius=2,
    top_m=4,
    temperature=0.07,
    lambda_dist=0.01,
    mode="soft",
)
```

第一版建议顺序：

1. `hard`
2. `soft`

transition feature 先只用：

```python
delta_i = matched - patch_t_i
delta_i = l2_normalize(delta_i)
```

这样仍可复用 same-grid temporal 的 whitening / likelihood 框架。

注意：

- 这是最复杂的一步
- 一定放在 same-grid temporal 跑通后再做

---

### 阶段 I：实现 evidence 输出

目标：

- 输出最异常 patch 的可解释证据

在 `PatchSTALL` 中输出：

```json
{
  "frame_id": 8,
  "patch_id": 72,
  "bbox_norm": [x1, y1, x2, y2],
  "branch": "patch_temporal",
  "ll": -1234.5
}
```

建议先输出 JSON，不接 MLLM。

---

## 3. 推荐实验顺序

建议不要一次跑满，而是按下面 ablation 顺序验证：

### A. 原版 baseline

```bash
python src/eval.py \
  --csv cache/indexes/comgenvid.csv \
  --emb-cache cache/embeddings/comgenvid \
  --duration 2 \
  --compact \
  --output-csv results/comgenvid_baseline.csv
```

### B. patch embedding debug

```bash
python src/eval_patch.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid_debug \
  --duration 2 \
  --compact \
  --debug 5 \
  --video-batch 2
```

### C. patch params（real-only）

```bash
python src/create_patch_params.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --output precomputed/patch_params_comgenvid_real.npz \
  --duration 2 \
  --compact \
  --real-only \
  --max-patches-for-fit 300000
```

### D. patch spatial only

```bash
python src/eval_patch.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --patch-params precomputed/patch_params_comgenvid_real.npz \
  --duration 2 \
  --compact \
  --fusion patch_only \
  --disable-patch-temp \
  --output-csv results/comgenvid_patch_spatial.csv
```

### E. patch spatial + same-grid temporal

```bash
python src/eval_patch.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --patch-params precomputed/patch_params_comgenvid_real.npz \
  --duration 2 \
  --compact \
  --fusion patch_only \
  --patch-temp-mode same_grid \
  --output-csv results/comgenvid_patch_samegrid.csv
```

### F. patch spatial + motion-soft temporal

```bash
python src/eval_patch.py \
  --csv cache/indexes/comgenvid.csv \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --patch-params precomputed/patch_params_comgenvid_real.npz \
  --duration 2 \
  --compact \
  --fusion patch_only \
  --patch-temp-mode motion_soft \
  --output-csv results/comgenvid_patch_motion_soft.csv
```

### G. global + patch fusion

```bash
python src/eval_patch.py \
  --csv cache/indexes/comgenvid.csv \
  --emb-cache cache/embeddings/comgenvid \
  --patch-emb-cache cache/patch_embeddings/comgenvid \
  --global-params precomputed/stall_params_vatex_dino_v3.npz \
  --patch-params precomputed/patch_params_comgenvid_real.npz \
  --duration 2 \
  --compact \
  --fusion avg \
  --patch-temp-mode motion_soft \
  --output-csv results/comgenvid_global_patch_fusion.csv
```

---

## 4. 关键注意事项

1. fake 视频不能参与真实分布拟合。
2. patch-level 分支必须重新拟合 patch params。
3. 原版 global 分支可以继续使用官方 VATEX params。
4. 第一版固定 `224x224`、`14x14` patch grid。
5. 第一版优先：
   - patch spatial
   - same-grid temporal
   - bottom-k 聚合
6. motion-soft matching 放在第二阶段。
7. 所有新增脚本都要支持 `--debug N`。
8. patch cache 体积会明显变大，优先先用：
   - `duration=2`
   - `compact=True`

---

## 5. 当前建议的第一步

当前最推荐先做的不是全改代码，而是：

1. 实现 `tools/inspect_dinov3_tokens.py`
2. 确认 patch token key 和 shape
3. 再开始写 `src/stall_patch.py` 的 patch/global embedding 抽取

只有这一步确认无误，后面的 patch cache、patch params、patch eval 才值得继续推进。
