"""相同目标真实信息下的二维(TTR,LSMI)Gaussian控制；复用已提取轻资产。"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import atomic_csv, checkpoint_read, paper_json
from config import config_digest
from reference import (
    file_digest,
    fit_gaussian,
    local_video_cdfs,
    percentile,
    publish_arrays,
    window_mean,
)
from evaluation.direction_study import fit_jobs
from evaluation.direction_analysis import shared_contrasts
from evaluation.tables import evaluate_fixed_pairs


def score(x, m):
    x = np.asarray(x, dtype=np.float64)
    return -0.5 * (np.square((x - m.mean) @ m.whitening).sum(-1) + 2 * np.log(2 * np.pi))


def run(root):
    source = root / "results/runs/local_direction_evidence"
    out = source / "scalar_control"
    out.mkdir(exist_ok=True)
    cfg = json.loads((source / "identity.json").read_text())
    identity = config_digest(cfg)
    c = cfg["config"]
    evaluated = json.loads((source / "evaluation.json").read_text())
    for p, h in evaluated["files"].items():
        if file_digest(source / p) != h:
            raise ValueError("原方向评价产物改变")
    did = json.loads((source / "dense_identity.json").read_text())["identity"]
    models = {}
    fits = {}
    for d in ("comgenvid", "videofeedback", "genvideo"):
        videos = []
        provenance = []
        for j in fit_jobs(root, c):
            if j["dataset"] != d:
                continue
            p = source / "fit_features" / (j["key"] + ".npz")
            r = checkpoint_read(p.with_suffix(".json"), identity, j["key"])
            if file_digest(p) != r["sha256"]:
                raise ValueError("标量拟合资产改变")
            with np.load(p) as z:
                videos.append(z["scalar"][:, :2].copy())
            provenance.append(
                dict(
                    video_id=j["row"]["video_id"],
                    source_group=j["row"]["source_group"],
                    sha256=r["sha256"],
                )
            )
        if len(videos) != 200:
            raise ValueError("目标真实拟合预算不同")
        models[d] = fit_gaussian(videos, ridge=1e-5)
        fits[d] = provenance
    arrays = {}
    for d, m in models.items():
        arrays[d + "_mean"] = m.mean
        arrays[d + "_whitening"] = m.whitening
    publish_arrays(out / "gaussians.npz", arrays)
    jobs = json.loads((root / c["plans"]).read_text())["jobs"]
    reference = {}
    raw = {}
    for j in jobs:
        r = checkpoint_read(source / "raw" / (j["key"] + ".json"), did, j["key"])
        scalar = np.asarray(r["scalars"])[:, :2]
        for d in j["targets"]:
            q = score(scalar, models[d])
            raw[j["key"], d] = q
            if j["role"] == "cdf":
                reference.setdefault((d, len(j["windows"][0])), []).append(q)
    cdfs = {}
    for key, values in reference.items():
        if len(values) != 2000:
            raise ValueError("CDF真实视频身份不足")
        cdfs[key] = (
            local_video_cdfs(values) if key[1] == 16 else {1: np.sort(np.asarray(values)[:, 0])}
        )
    baseline = pd.read_csv(source / "video_scores.csv.gz", float_precision="round_trip")
    glob = baseline[baseline.variant.eq("global_only")].set_index("video_id")
    rows = []
    for j in jobs:
        if j["role"] != "evaluation":
            continue
        length = len(j["windows"][0])
        k = len(j["windows"])
        for d, m in j["targets"].items():
            q = window_mean(raw[j["key"], d])
            l = float(percentile([q], cdfs[d, length][k])[0])
            g = float(glob.loc[m["video_id"], "final_score"])
            meta = {
                n: m[n] for n in ("video_id", "dataset", "subset", "source_model", "source_group")
            }
            meta.update(video_path=j["video_path"], effective_k=k, window_frames=length)
            for suffix, value in (("", 0.5 * g + 0.5 * l), ("_only", l), ("_raw", q)):
                rows.append(dict(**meta, variant="scalar_gaussian" + suffix, final_score=value))
    s = pd.DataFrame(rows)
    s.to_csv(out / "video_scores.csv.gz", index=False)
    pairs = pd.read_csv(root / c["baseline"] / "pairs.csv")
    tables = {}
    for name, part in s.groupby("variant"):
        for key, t in evaluate_fixed_pairs(part, pairs).items():
            tables.setdefault(key, []).append(t.assign(variant=name))
    for key, items in tables.items():
        atomic_csv(
            out / (key + ".csv"), pd.concat(items).replace({"scope": {"Macro-3": "Average"}})
        )
    groups = s[["video_id", "source_group"]].drop_duplicates().set_index("video_id").source_group
    frames = {
        n: baseline[baseline.variant.eq(n)]
        for n in ("d2_anchor", "d2_anchor_raw", "split", "split_raw")
    }
    frames.update({n: s[s.variant.eq(n)] for n in ("scalar_gaussian", "scalar_gaussian_raw")})
    contrasts = {
        "D2_vs_scalar_Gaussian": {"d2_anchor": 1, "scalar_gaussian": -1},
        "D2raw_vs_scalar_Gaussian_raw": {"d2_anchor_raw": 1, "scalar_gaussian_raw": -1},
        "scalar_Gaussian_vs_SPLIT": {"scalar_gaussian": 1, "split": -1},
    }
    ci = shared_contrasts(frames, pairs, groups, contrasts, iterations=1000, seed=17)
    atomic_csv(out / "contrasts.csv", ci)
    publish_arrays(
        out / "cdfs.npz",
        {f"{d}__T{t}__K{k}": v for (d, t), items in cdfs.items() for k, v in items.items()},
    )
    paper_json(
        out / "manifest.json",
        dict(
            status="completed",
            protocol="target_scalar_statistics_control_v1",
            source_identity=file_digest(source / "identity.json"),
            source_scores=file_digest(source / "video_scores.csv.gz"),
            code=file_digest(Path(__file__)),
            fit=fits,
            fit_observations="per-window TTR/LSMI, equal total weight per fit clip",
            cdf="same independent VATEX2000, own Gaussian, matched 8/16 and effective-K",
            dimensions=2,
            ridge=1e-5,
            score_direction="higher Gaussian score is real; no test-dependent flipping",
            files={p.name: file_digest(p) for p in out.iterdir() if p.is_file()},
        ),
    )
    print(
        pd.read_csv(out / "macro_metrics.csv")[["variant", "auc", "real_positive_ap"]].to_string(
            index=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--wait-pid", type=int, help="等待已绑定评分进程退出；不启动新的GPU任务")
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
                raise RuntimeError("请在评分完成后运行此控制")
            if fd < 0:
                raise OSError(ctypes.get_errno(), "无法绑定评分进程")
        try:
            while not select.select([fd], [], [], 30)[0]:
                pass
        finally:
            os.close(fd)
        state = json.loads(
            (a.root / "results/runs/local_direction_evidence/status.json").read_text()
        )
        if state.get("status") != "scores_complete":
            raise RuntimeError("评分未完整结束")
    run(a.root.resolve())
