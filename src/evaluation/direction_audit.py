"""方向研究独立复核：完整身份、CDF映射、指标及分层Patch直接公式。"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

from artifacts import checkpoint_read, paper_json
from config import config_digest
from reference import file_digest, window_mean


def independent_statistics(patches):
    """以逐窗口直接公式重算，不调用研究的特征或评分函数。"""
    out = []
    directions = []
    for values in patches:
        x = torch.from_numpy(np.asarray(values, dtype=np.float32))
        t, n, d = x.shape
        h = int(n**0.5)
        v = x[1:] - x[:-1]
        two = x[2:] - x[:-2]
        a = torch.linalg.vector_norm(v, dim=-1).sum(0)
        b = torch.linalg.vector_norm(two, dim=-1).sum(0) / 2 * (t - 1) / (t - 2)
        rough = ((torch.log(a + 1e-8) - torch.log(b + 1e-8)) / 0.693147).mean()
        v = v.reshape(t - 1, h, h, d)
        horizontal = torch.linalg.vector_norm(v[:, :, :-1] - v[:, :, 1:], dim=-1).mean()
        vertical = torch.linalg.vector_norm(v[:, :-1] - v[:, 1:], dim=-1).mean()
        spatial = (horizontal + vertical) / 2
        out.append([float(rough), float(spatial), float(rough**8 * spatial)])
        mean = x.mean(1)
        raw = mean[2:] - 2 * mean[1:-1] + mean[:-2]
        directions.append(torch.nn.functional.normalize(raw, dim=-1, eps=1e-12).numpy())
    return np.asarray(out), np.stack(directions)


def audit(root):
    out = root / "results/runs/local_direction_evidence"
    spec = json.loads((out / "identity.json").read_text())
    identity = config_digest(spec)
    for p, h in spec["inputs"].items():
        if file_digest(root / p) != h:
            raise ValueError("冻结来源改变：" + p)
    for p, h in spec["code"].items():
        if file_digest(root / p) != h:
            raise ValueError("冻结数值代码改变：" + p)
    dmeta = json.loads((out / "dense_identity.json").read_text())
    did = dmeta["identity"]
    for d, h in dmeta["models"].items():
        if file_digest(out / "models" / f"{d}.npz") != h:
            raise ValueError("评分模型改变")
        model = checkpoint_read(out / "models" / f"{d}.json", identity, d)
        if model["sha256"] != h or len(model["fit"]) != 200:
            raise ValueError("拟合预算/模型身份不符")
        if set(model["draws"]) != {"17", "29", "43", "59", "71"}:
            raise ValueError("源重抽样数量不符")
        for draw in model["draws"].values():
            if sum(draw.values()) != len({r["source_group"] for r in model["fit"]}):
                raise ValueError("未以源组为单位重抽样")
    evaluation = json.loads((out / "evaluation.json").read_text())
    for p, h in evaluation["files"].items():
        if file_digest(out / p) != h:
            raise ValueError("评价产物改变")
    base = root / "results/paper_complete"
    meta = pd.read_csv(base / "evaluation.csv", keep_default_na=False)
    pairs = pd.read_csv(base / "pairs.csv", keep_default_na=False)
    scores = pd.read_csv(out / "video_scores.csv.gz", float_precision="round_trip")
    wanted = set(meta.video_id)
    names = set(scores.variant)
    if len(wanted) != 15569 or len(names) != 70 or scores.duplicated(["variant", "video_id"]).any():
        raise ValueError("评分数量或唯一性错误")
    for name, g in scores.groupby("variant"):
        if set(g.video_id) != wanted or not np.isfinite(g.final_score).all():
            raise ValueError("评分缺失或非有限")
    old = pd.read_csv(base / "scores_fc3.csv.gz", float_precision="round_trip")
    compare = []
    for new, prior in [
        ("global_only", "global_only"),
        ("d2_anchor", "full"),
        ("d1_anchor", "local_d1_target"),
        ("d2_diagonal", "local_d2_target_diagonal"),
    ]:
        a = scores[scores.variant.eq(new)].set_index("video_id").final_score
        b = old[old.variant.eq(prior)].set_index("video_id").final_score.loc[a.index]
        delta = np.abs(a.to_numpy() - b.to_numpy())
        if new in ("global_only", "d2_anchor") and delta.max() > 1e-12:
            raise ValueError("Global/D2锚点改变")
        compare.append(
            dict(
                variant=new,
                max_final_difference=float(delta.max()),
                nonzero_rows=int((delta > 1e-12).sum()),
            )
        )
    # 不调用evaluate_fixed_pairs，直接按冻结pair身份由sklearn重建全部单元指标。
    printed = pd.read_csv(out / "generator_metrics.csv").set_index(
        ["variant", "dataset", "generator"]
    )
    max_metric = 0.0
    for name, part in scores.groupby("variant"):
        indexed = part.set_index("video_id")
        for (d, g), group in pairs.groupby(["dataset", "generator"]):
            s = indexed.loc[group.video_id]
            y = group.subset.eq("real").to_numpy()
            actual = np.array(
                [roc_auc_score(y, s.final_score), average_precision_score(y, s.final_score)]
            )
            expected = printed.loc[name, d, g][["auc", "real_positive_ap"]].to_numpy(dtype=float)
            max_metric = max(max_metric, float(np.abs(actual - expected).max()))
    if max_metric > 1e-12:
        raise ValueError("独立指标复核不符")
    jobs = json.loads((root / spec["config"]["plans"]).read_text())["jobs"]
    selected = []
    counts = Counter()
    for j in jobs:
        if j["role"] != "evaluation":
            continue
        d = next(iter(j["targets"]))
        m = j["targets"][d]
        tag = (d, len(j["windows"][0]), m["subset"])
        if counts[tag] < 2:
            selected.append(j)
            counts[tag] += 1
    probes = []
    cache = root / spec["config"]["patch_cache"]
    lookup = scores.set_index(["variant", "video_id"])
    reference_rows = {}
    for j in jobs:
        if j["role"] != "cdf":
            continue
        record = checkpoint_read(out / "raw" / (j["key"] + ".json"), did, j["key"])
        length = len(j["windows"][0])
        for d, entries in record["targets"].items():
            values = dict(entries)
            for i, name in enumerate(("ttr", "lsmi", "split")):
                values[name] = (-np.asarray(record["scalars"])[:, i]).tolist()
            for name, v in values.items():
                reference_rows.setdefault((d, length, name), []).append(v)
    with np.load(out / "cdf_arrays.npz") as cdfs:
        checked_arrays = 0
        for (d, length, name), videos in reference_rows.items():
            if len(videos) != 2000:
                raise ValueError("独立重建CDF参考身份数不足2000")
            for k in (1,) if length == 8 else (1, 2, 3):
                rebuilt = []
                for values in videos:
                    if len(values) < k:
                        continue
                    positions = np.unique(np.rint(np.linspace(0, len(values) - 1, k)).astype(int))
                    rebuilt.append(float(np.asarray(values, dtype=np.float64)[positions].mean()))
                np.testing.assert_array_equal(
                    np.sort(rebuilt), cdfs[f"{d}__T{length}__{name}__K{k}"]
                )
                checked_arrays += 1
        if checked_arrays != len(cdfs.files):
            raise ValueError("CDF有遗漏或未登记的模型")
        for j in selected:
            r = json.loads((cache / (j["key"] + ".json")).read_text())
            p = cache / (j["key"] + ".npy")
            if file_digest(p) != r["sha256"]:
                raise ValueError("分层Patch hash改变")
            x = np.load(p)
            patch = np.stack([x[ix] for ix in r["window_positions"]])
            scalar, u = independent_statistics(patch)
            record = checkpoint_read(out / "raw" / (j["key"] + ".json"), did, j["key"])
            # 独立归约次序的FP32差异使用明确相对容差，不放宽主路径D2回归。
            np.testing.assert_allclose(scalar, record["scalars"], rtol=1e-5, atol=2e-6)
            for d, m in j["targets"].items():
                with np.load(out / "models" / f"{d}.npz") as z:
                    mean = z["pooled_d2_mean"]
                    W = z["pooled_d2_whitening"]
                    q = -0.5 * (
                        (((u.astype(float) - mean) @ W) ** 2).sum(-1)
                        + len(mean) * np.log(2 * np.pi)
                    ).mean(1)
                np.testing.assert_allclose(q, record["targets"][d]["pooled_d2"], rtol=0, atol=1e-10)
                all_values = dict(record["targets"][d])
                for i, n in enumerate(("ttr", "lsmi", "split")):
                    all_values[n] = (-np.asarray(record["scalars"])[:, i]).tolist()
                for name, values in all_values.items():
                    raw = window_mean(values)
                    a = cdfs[f"{d}__T{len(j['windows'][0])}__{name}__K{len(j['windows'])}"]
                    l = float(np.count_nonzero(a <= raw) / len(a))
                    g = float(lookup.loc["global_only", m["video_id"]].final_score)
                    actual = [raw, l, 0.5 * g + 0.5 * l]
                    expected = [
                        float(lookup.loc[name + s, m["video_id"]].final_score)
                        for s in ("_raw", "_only", "")
                    ]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            probes.append(
                dict(
                    key=j["key"],
                    windows=len(j["windows"]),
                    frames=len(j["windows"][0]),
                    sha256=r["sha256"],
                )
            )
    receipt = dict(
        status="passed",
        clips=15569,
        variants=len(names),
        cells=23,
        direct_metric_rows=len(printed),
        max_metric_error=max_metric,
        baseline_regression=compare,
        independently_rebuilt_cdf_arrays=checked_arrays,
        probes=probes,
        source_score_sha256=file_digest(out / "video_scores.csv.gz"),
        auditor_sha256=file_digest(Path(__file__)),
        limitations=[
            "reference bootstrap remains conditional on original real pool",
            "probe numerical check is stratified, not second full Patch hash scan",
        ],
    )
    paper_json(out / "independent_audit.json", receipt)
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "probes"}, ensure_ascii=False, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--wait-pid", type=int, help="绑定已运行的后处理进程，结束后才独立复核")
    a = p.parse_args()
    if a.wait_pid is not None:
        import ctypes, os, select

        if hasattr(os, "pidfd_open"):
            fd = os.pidfd_open(a.wait_pid)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            if hasattr(libc, "pidfd_open"):
                fd = libc.pidfd_open(a.wait_pid, 0)
            elif os.uname().machine == "x86_64":
                fd = libc.syscall(434, a.wait_pid, 0)
            else:
                raise RuntimeError("无pidfd接口，请后处理完成后直接运行audit")
            if fd < 0:
                raise OSError(ctypes.get_errno(), "无法绑定后处理进程")
        try:
            while not select.select([fd], [], [], 30)[0]:
                pass
        finally:
            os.close(fd)
        marker = a.root / "results/runs/local_direction_evidence/verification.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "verified":
            raise RuntimeError("后处理未完整完成，不生成独立验收结论")
    torch.set_num_threads(2)
    audit(a.root.resolve())
