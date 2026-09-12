"""按完整域提前导出新参考结果；五次指标平均与集成预测严格区分。"""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from artifacts import atomic_csv, paper_json
from config import config_digest
from evaluation.confirmation_data import context, DOMAINS
from evaluation.confirmation_engine import Chunks
from evaluation.tables import evaluate_fixed_pairs
from evaluation.direction_analysis import shared_contrasts
from reference import percentile, window_mean, local_video_cdfs, file_digest


def collect(root):
    c, out, data = context(root)
    spec = json.loads((out / "score_identity.json").read_text())
    identity = config_digest(spec)
    records = {}
    for p in (out / "cdf_rank0", out / "cdf_rank1", out / "evaluation_chunks"):
        store = Chunks(p, identity)
        if set(records) & set(store.records):
            raise ValueError("跨分片视频重复")
        records.update(store.records)
    jobs = json.loads((root / c["plans"]).read_text())["jobs"]
    return c, out, identity, jobs, records


def export(root, partial=False):
    c, out, identity, jobs, records = collect(root)
    complete = [d for d in DOMAINS if all(j["key"] in records for j in jobs if d in j["targets"])]
    if not partial and len(complete) != 3:
        raise ValueError("评分未覆盖三域")
    if not complete:
        return
    dest = out / ("partial" if partial else "results")
    dest.mkdir(exist_ok=True)
    previous = dest / "manifest.json"
    if previous.exists() and json.loads(previous.read_text())["domains"] == complete:
        return
    reference = {}
    global_reference = {}
    for j in jobs:
        if j["role"] != "cdf" or j["key"] not in records:
            continue
        t = len(j["windows"][0])
        r = records[j["key"]]
        for d, v in r["targets"].items():
            if d not in complete:
                continue
            for tag, g in v["global_values"].items():
                for b in ("gs", "gt"):
                    global_reference.setdefault((d, t, tag, b), []).append(float(g["uniform"][b]))
            for tag, q in v["local"].items():
                reference.setdefault((d, t, tag), []).append(q)
    local_cdf = {}
    for key, values in reference.items():
        if len(values) != 2000:
            raise ValueError("参考模型对应CDF不足2000")
        local_cdf[key] = (
            local_video_cdfs(values) if key[1] == 16 else {1: np.sort(np.asarray(values)[:, 0])}
        )
    if any(len(v) != 2000 for v in global_reference.values()):
        raise ValueError("Global对应CDF不足2000")
    rows = []
    for j in jobs:
        if j["role"] != "evaluation" or j["key"] not in records:
            continue
        t = len(j["windows"][0])
        k = len(j["windows"])
        r = records[j["key"]]
        for d, meta in j["targets"].items():
            if d not in complete:
                continue
            v = r["targets"][d]
            glob = {}
            loc = {}
            raw = {}
            for tag, g in v["global_values"].items():
                gs = percentile([float(x) for x in g["gs"]], global_reference[d, t, tag, "gs"])
                gt = percentile(
                    [float(x) for x in g["gt"]],
                    global_reference[d, t, tag, "gt"],
                    allow_positive_infinity=True,
                )
                glob[tag] = window_mean(0.5 * gs + 0.5 * gt)
            for tag, q in v["local"].items():
                raw[tag] = window_mean(q)
                loc[tag] = float(percentile([raw[tag]], local_cdf[d, t, tag][k])[0])
            values = dict(
                original_global=glob["anchor"],
                original_full=0.5 * glob["anchor"] + 0.5 * loc["anchor"],
                original_matched=0.5 * glob["anchor"] + 0.5 * loc["matched_anchor"],
                original_matched_raw=raw["matched_anchor"],
                original_raw=raw["anchor"],
            )
            for seed in c["seeds"]:
                tag = f"s{seed}"
                g = glob[tag]
                values[tag + "_global"] = g
                for name, local in [
                    ("full", "d2"),
                    ("source", "source_equal"),
                    ("d1", "d1"),
                    ("matched", "matched"),
                ]:
                    values[tag + "_" + name] = 0.5 * g + 0.5 * loc[tag + "_" + local]
            fields = {
                p: meta[p]
                for p in ("video_id", "dataset", "subset", "source_model", "source_group")
            }
            fields.update(window_frames=t, effective_k=k)
            rows.extend(
                [dict(**fields, variant=name, final_score=value) for name, value in values.items()]
            )
    scores = pd.DataFrame(rows)
    base = pd.read_csv(root / c["baseline"] / "scores_fc3.csv.gz", float_precision="round_trip")
    for new, old in [("original_global", "global_only"), ("original_full", "full")]:
        a = scores[scores.variant == new].set_index("video_id").final_score
        b = base[base.variant == old].set_index("video_id").final_score.loc[a.index]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    pairs = pd.read_csv(root / c["baseline"] / "pairs.csv")
    pairs = pairs[pairs.dataset.isin(complete)]
    tables = {}
    for name, part in scores.groupby("variant"):
        for key, t in evaluate_fixed_pairs(part, pairs).items():
            tables.setdefault(key, []).append(t.assign(variant=name))
    path = dest / "video_scores.csv.gz"
    temp = dest / "video_scores.tmp.csv.gz"
    scores.to_csv(temp, index=False, compression="gzip")
    temp.replace(path)
    for name, t in tables.items():
        atomic_csv(dest / (name + ".csv"), pd.concat(t).replace({"scope": {"Macro-3": "Average"}}))
    means = []
    metrics = pd.read_csv(dest / "dataset_metrics.csv")
    if len(complete) == 3:
        metrics = pd.concat(
            [metrics, pd.read_csv(dest / "macro_metrics.csv").rename(columns={"scope": "dataset"})]
        )
    for d, group in metrics.groupby("dataset"):
        for v in ("global", "full", "source", "d1", "matched"):
            q = group[group.variant.isin([f"s{s}_{v}" for s in c["seeds"]])]
            if len(q) != 5:
                raise ValueError("新池重复不足五次")
            means.append(
                dict(
                    dataset=d,
                    variant=v,
                    auc_mean=q.auc.mean(),
                    auc_sd=q.auc.std(ddof=1),
                    ap_mean=q.real_positive_ap.mean(),
                    ap_sd=q.real_positive_ap.std(ddof=1),
                )
            )
    atomic_csv(dest / "reference_means.csv", pd.DataFrame(means))
    paper_json(
        previous,
        dict(
            status="complete" if len(complete) == 3 else "complete_domains_only",
            domains=complete,
            identity=identity,
            average="mean of five per-model metrics, not metric of averaged predictions",
            files={
                p.name: file_digest(p)
                for p in dest.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        ),
    )
    print(pd.DataFrame(means).to_string(index=False), flush=True)


def analyze(root):
    c, out, data = context(root)
    dest = out / "results"
    m = json.loads((dest / "manifest.json").read_text())
    if m["status"] != "complete":
        raise ValueError("不能分析不完整域")
    for p, h in m["files"].items():
        if file_digest(dest / p) != h:
            raise ValueError("新参考结果改变")
    s = pd.read_csv(dest / "video_scores.csv.gz", float_precision="round_trip")
    pairs = pd.read_csv(root / c["baseline"] / "pairs.csv")
    groups = s[["video_id", "source_group"]].drop_duplicates().set_index("video_id").source_group
    contrasts = {}
    for name, a, b in [
        ("new_Full_vs_Global", "full", "global"),
        ("new_source_vs_clip", "source", "full"),
        ("new_D2_vs_D1", "full", "d1"),
        ("new_full_vs_position_matched", "full", "matched"),
    ]:
        coeff = {}
        for seed in c["seeds"]:
            coeff[f"s{seed}_{a}"] = 0.2
            coeff[f"s{seed}_{b}"] = -0.2
        contrasts[name] = coeff
    contrasts["original_full_vs_position_matched"] = {"original_full": 1, "original_matched": -1}
    previous = pd.read_csv(
        root / "results/runs/local_direction_evidence/video_scores.csv.gz",
        float_precision="round_trip",
    )
    contrasts["position_matched_vs_pooled"] = {"original_matched": 1, "pooled_d2": -1}
    contrasts["matched_raw_vs_pooled_raw"] = {"original_matched_raw": 1, "pooled_d2_raw": -1}
    names = {n for v in contrasts.values() for n in v}
    frames = {
        n: s[s.variant == n] if n in set(s.variant) else previous[previous.variant == n]
        for n in names
    }
    ci = shared_contrasts(
        frames, pairs, groups, contrasts, iterations=c["bootstrap_iterations"], seed=17
    )
    atomic_csv(dest / "contrasts.csv", ci)
    paper_json(
        dest / "analysis.json",
        dict(
            status="completed",
            contrasts=contrasts,
            uncertainty="evaluation source bootstrap conditional on five frozen new fit pools; fit-pool variability separately reported",
            source=file_digest(dest / "video_scores.csv.gz"),
            code=file_digest(Path(__file__)),
            output=file_digest(dest / "contrasts.csv"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["export", "analyze"])
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    if a.action == "export":
        export(Path.cwd(), a.partial)
    else:
        analyze(Path.cwd())
