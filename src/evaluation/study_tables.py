"""由完整、匹配模型的共享证据生成论文对照；禁止从旧表补分数。"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import checkpoint_read, paper_json
from config import config_digest
from reference import local_video_cdfs, percentile, window_mean, file_digest
from branches.global_branch import load_official_stall_parameters
from evaluation.tables import evaluate_fixed_pairs


def evidence_ready(directory):
    directory = Path(directory)
    first = directory / "completed_rank_0.json"
    if not first.exists():
        return False
    world_size = json.loads(first.read_text())["world_size"]
    if type(world_size) is not int or not 1 <= world_size <= 64:
        raise ValueError("分片数量非法")
    return all((directory / f"completed_rank_{r}.json").exists() for r in range(world_size))


def read_evidence(directory, frame, world_size=None):
    directory = Path(directory)
    identities = {}
    if world_size is None:
        world_size = json.loads((directory / "completed_rank_0.json").read_text())["world_size"]
    if type(world_size) is not int or not 1 <= world_size <= len(frame):
        raise ValueError("分片数量非法")
    for rank in range(world_size):
        spec = json.loads((directory / f"identity_rank_{rank}.json").read_text())
        done = json.loads((directory / f"completed_rank_{rank}.json").read_text())
        identity = config_digest(spec)
        if done["identity"] != identity or done["world_size"] != world_size:
            raise ValueError("证据完成身份不符")
        if spec["manifest_sha256"] != config_digest(frame.to_dict("records")):
            raise ValueError("证据清单不符")
        if rank:
            previous = identities[0][1]
            for key in ("models", "include_uniform", "code"):
                if spec.get(key) != previous.get(key):
                    raise ValueError("不同分片的模型或实现身份不一致")
            if "config" in spec:

                def canonical(value):
                    value = json.loads(json.dumps(value))
                    value.get("runtime", {}).pop("device", None)
                    return value

                if canonical(spec["config"]) != canonical(previous["config"]):
                    raise ValueError("不同分片的配置不一致")
        identities[rank] = (identity, spec)
    records = []
    for i, row in enumerate(frame.itertuples()):
        records.append(
            checkpoint_read(
                directory / "raw" / f"{i:06d}.json", identities[i % world_size][0], row.video_id
            )["evidence"]
        )
    return records, identities[0][1]


def calibrated_scores(
    root, domain, frame, evaluation, calibration, requested_k, baseline_only=False
):
    """评价和独立阈值共用完全相同的映射与融合；调用方先验收身份。"""
    root = Path(root)
    if len(frame) != len(evaluation) or not calibration:
        raise ValueError("评分身份数量或参考为空")
    gc = {}
    for model in ("target",) if baseline_only else ("target", "source"):
        gc[model] = {
            name: np.asarray(
                [float(x["uniform"]["global_models"][model][name]) for x in calibration]
            )
            for name in ("gs", "gt")
        }
    if not baseline_only:
        official = load_official_stall_parameters(
            root / "precomputed/stall_params_vatex_dino_v3.npz"
        )
        gc["official"] = dict(
            gs=official["global_spatial"].calibration_raw, gt=official["global_t1"].calibration_raw
        )
    first = calibration[0]["selected"]["windows"][0]["representations"]
    lc = {}
    for representation, models in first.items():
        for model in models:
            ranked = [
                [
                    float(w["representations"][representation][model])
                    for w in x["selected"]["windows"]
                ]
                for x in calibration
            ]
            if requested_k == 1:
                if any(len(x) != 1 for x in ranked):
                    raise ValueError("K1参考包含多个窗口")
                lc[representation, model] = {1: np.sort(np.asarray([x[0] for x in ranked]))}
            else:
                lc[representation, model] = local_video_cdfs(ranked)
    rows = []
    for row, raw in zip(frame.to_dict("records"), evaluation):
        windows = raw["windows"]
        k = len(windows)
        g = {}
        parts = {}
        local = {}
        q = {}
        for model in gc:
            gs = percentile(
                [float(w["global_models"][model]["gs"]) for w in windows], gc[model]["gs"]
            )
            gt = percentile(
                [float(w["global_models"][model]["gt"]) for w in windows],
                gc[model]["gt"],
                allow_positive_infinity=True,
            )
            g[model] = window_mean(0.5 * gs + 0.5 * gt)
            parts[model] = (window_mean(gs), window_mean(gt))
        for key, cdfs in lc.items():
            r, m = key
            q[key] = window_mean([float(w["representations"][r][m]) for w in windows])
            local[key] = float(percentile([q[key]], cdfs[k])[0])
        l = local["local_d2", "target"]
        variants = dict(
            full=0.5 * g["target"] + 0.5 * l,
            global_only=g["target"],
            local_only=l,
            without_gs=0.5 * parts["target"][1] + 0.5 * l,
            without_gt=0.5 * parts["target"][0] + 0.5 * l,
        )
        if not baseline_only:
            variants["official_global"] = g["official"]
        for gm in () if baseline_only else ("source", "target"):
            for lm in ("source", "target"):
                variants[f"global_{gm}_local_{lm}"] = 0.5 * g[gm] + 0.5 * local["local_d2", lm]
        for key, value in local.items():
            r, m = key
            variants[f"{r}_{m}"] = 0.5 * g["target"] + 0.5 * value
            variants[f"{r}_{m}_only"] = value
            variants[f"{r}_{m}_raw"] = q[key]
        for name, value in variants.items():
            rows.append(
                dict(
                    video_id=row["video_id"],
                    dataset=domain,
                    subset=row["subset"],
                    source_model=row["source_model"],
                    video_path=row["video_path"],
                    variant=name,
                    final_score=value,
                    effective_k=k,
                )
            )
    return pd.DataFrame(rows)


def aggregate_domain(root, domain, evaluation_dir, cdf_dir, output):
    root = Path(root)
    output = Path(output)
    frame = pd.read_csv(
        root / "data/manifests/active" / domain / "evaluation.csv", keep_default_na=False
    )
    cframe = pd.read_csv(root / "data/manifests/active/vatex/cdf.csv", keep_default_na=False)
    evaluation, ei = read_evidence(evaluation_dir, frame)
    calibration, ci = read_evidence(cdf_dir, cframe)
    if ei["models"] != ci["models"]:
        raise ValueError("CDF与评价Gaussian不同")
    if ei["config"]["selection"] != ci["config"]["selection"]:
        raise ValueError("CDF与评价观察策略不同")
    if ei["config"].get("evidence_scope") != ci["config"].get("evidence_scope"):
        raise ValueError("CDF与评价证据范围不同")
    scores = calibrated_scores(
        root,
        domain,
        frame,
        evaluation,
        calibration,
        ci["config"]["selection"]["k"],
        ci["config"].get("evidence_scope") == "baseline_budget",
    )
    pairs = pd.read_csv(root / "data/manifests/active" / domain / "pairs.csv")
    tables = {}
    for variant, part in scores.groupby("variant", sort=False):
        for name, table in evaluate_fixed_pairs(part, pairs).items():
            tables.setdefault(name, []).append(table.assign(variant=variant))
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "video_scores.csv.gz", index=False)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            dataset=domain,
            evaluation=str(evaluation_dir),
            cdf=str(cdf_dir),
            models=ei["models"],
            selection=ei["config"]["selection"],
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
            metric_policy="round-trip CSV; anomaly=-score; right-inclusive CDF; discrete ROC with all thresholds",
            note="点估计；bootstrap、跨域Macro、阈值迁移另行计算",
        ),
    )
    return scores


def aggregate_official(root, domain, directory, output, world_size=2):
    root = Path(root)
    directory = Path(directory)
    output = Path(output)
    frame = pd.read_csv(
        root / "data/manifests/active" / domain / "evaluation.csv", keep_default_na=False
    )
    identities = {}
    for rank in range(world_size):
        spec = json.loads((directory / f"identity_rank_{rank}.json").read_text())
        done = json.loads((directory / f"completed_rank_{rank}.json").read_text())
        identity = config_digest(spec)
        if done["identity"] != identity or spec["manifest_sha256"] != config_digest(
            frame.to_dict("records")
        ):
            raise ValueError("官方单窗身份或完成状态错误")
        identities[rank] = identity
    rows = []
    for i, row in enumerate(frame.to_dict("records")):
        payload = checkpoint_read(
            directory / "raw" / f"{i:06d}.json", identities[i % world_size], row["video_id"]
        )
        if payload["frame_indices"] != json.loads(row["2_sec_idxs"]):
            raise ValueError("官方窗口身份漂移")
        rows.append(
            {
                **{
                    k: row[k]
                    for k in ("video_id", "dataset", "subset", "source_model", "video_path")
                },
                "final_score": float(payload["final_score"]),
                "variant": "official_single_window",
            }
        )
    scores = pd.DataFrame(rows)
    pairs = pd.read_csv(root / "data/manifests/active" / domain / "pairs.csv")
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "video_scores.csv.gz", index=False)
    for name, table in evaluate_fixed_pairs(scores, pairs).items():
        table.to_csv(output / (name + ".csv"), index=False)
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            dataset=domain,
            source=str(directory),
            variant="official_single_window",
            metric_policy="round-trip CSV; anomaly=-score; discrete ROC with all thresholds",
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    return scores


def combine_domains(root, inputs, output):
    """三个域齐备后重新计算Macro，禁止按完整池视频数量加权。"""
    root = Path(root)
    output = Path(output)
    parts = []
    provenance = {}
    for directory in map(Path, inputs):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("表格run未完成")
        for name, digest in manifest["files"].items():
            if file_digest(directory / name) != digest:
                raise ValueError("域级结果文件改变")
        part = pd.read_csv(directory / "video_scores.csv.gz", float_precision="round_trip")
        parts.append(part)
        provenance[str(directory)] = file_digest(directory / "manifest.json")
    scores = pd.concat(parts, ignore_index=True)
    pairs = pd.read_csv(root / "data/manifests/active/pairs.csv")
    pairs = pairs[pairs.dataset.isin(("comgenvid", "videofeedback", "genvideo"))]
    tables = {}
    for variant, part in scores.groupby("variant", sort=False):
        if set(part.dataset) != {"comgenvid", "videofeedback", "genvideo"}:
            raise ValueError("变体未覆盖三个域")
        for name, table in evaluate_fixed_pairs(part, pairs).items():
            tables.setdefault(name, []).append(table.assign(variant=variant))
    output.mkdir(parents=True, exist_ok=False)
    scores.to_csv(output / "video_scores.csv.gz", index=False)
    for name, items in tables.items():
        pd.concat(items, ignore_index=True).to_csv(output / (name + ".csv"), index=False)
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            inputs=provenance,
            scope="three development datasets / fixed 20 cells",
            metric_policy="round-trip CSV; anomaly=-score; discrete ROC with all thresholds",
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    return scores
