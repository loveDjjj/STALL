"""独立单进程成本测量；使用实际D2主线，不计入共享消融评分器的额外计算。"""

import hashlib, json, time, gc, resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import psutil
from artifacts import paper_json
from reference import load_bundle, percentile, window_mean, file_digest
from workflow import RawVideoScorer
from evaluation.study_tables import read_evidence
from data.sequential_decode import SequentialDecodeMixin


def benchmark_domain(root, config, domain, output):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(
        root / "data/manifests/active" / domain / "evaluation.csv", keep_default_na=False
    )
    frame["order"] = frame.video_id.map(
        lambda x: hashlib.sha256(("runtime17:" + x).encode()).hexdigest()
    )
    sample = (
        frame.sort_values("order").groupby(["subset", "source_model"], sort=True).head(2).copy()
    )
    sample.to_csv(output / "sample.csv", index=False)
    reference_path = root / "results/runs" / f"paper_fit_{domain}" / "references" / f"{domain}.npz"
    reference = load_bundle(reference_path)
    device = config["runtime"]["device"]
    started = time.perf_counter()

    class SequentialScorer(SequentialDecodeMixin, RawVideoScorer):
        pass

    scorer = SequentialScorer(
        reference.global_spatial,
        reference.global_temporal,
        reference.local_temporal,
        device,
        dino_repo=str(root / config["encoder"]["repo"]),
        dino_weights=str(root / config["encoder"]["weights"]),
    )
    initialization = time.perf_counter() - started
    clock = dict(global_forward=0.0, dense_forward=0.0)

    def timed(fn, key):
        def call(*args, **kwargs):
            torch.cuda.synchronize(device)
            t = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize(device)
            clock[key] += time.perf_counter() - t
            return result

        return call

    scorer.extractor.frames_to_global_embeddings = timed(
        scorer.extractor.frames_to_global_embeddings, "global_forward"
    )
    scorer.extractor.frames_to_global_patch_embeddings = timed(
        scorer.extractor.frames_to_global_patch_embeddings, "dense_forward"
    )
    rows = []
    warmups = {}
    cdf_frame = pd.read_csv(root / "data/manifests/active/vatex/cdf.csv", keep_default_na=False)
    for name, selector, k in [
        ("uniform1", "uniform", 1),
        ("fc1", "feature_change", 1),
        ("uniform3", "uniform", 3),
        ("fc3", "feature_change", 3),
    ]:
        records, provenance = read_evidence(
            root / "results/runs" / f"paper_cdf_{domain}_{name}", cdf_frame
        )
        if reference.metadata["gaussian_sha256"] not in provenance["models"].values():
            raise ValueError("成本测试CDF与Gaussian不匹配")
        ranked = [
            [float(w["representations"]["local_d2"]["target"]) for w in x["selected"]["windows"]]
            for x in records
        ]
        cdfs = {}
        for effective in range(1, k + 1):
            cdfs[effective] = np.sort(
                [
                    np.mean(
                        np.asarray(x)[
                            np.unique(np.rint(np.linspace(0, len(x) - 1, effective)).astype(int))
                        ]
                    )
                    for x in ranked
                    if len(x) >= effective
                ]
            )
        del records
        gc.collect()
        warm = sample.sort_values("duration_seconds", ascending=False).iloc[0]
        t = time.perf_counter()
        scorer.score_raw(
            root / warm.video_path, json.loads(warm.downsample_idxs), selector=selector, k=k
        )
        torch.cuda.synchronize(device)
        warmups[name] = time.perf_counter() - t
        for repeat in (0, 1):
            for row in sample.itertuples():
                indices = json.loads(row.downsample_idxs)
                path = root / row.video_path
                torch.cuda.reset_peak_memory_stats(device)
                clock.update(global_forward=0.0, dense_forward=0.0)
                start = time.perf_counter()
                t = start
                prepared = scorer.prepare_coarse(
                    path, indices, selector=selector, k=k, materialize=True
                )
                coarse_cpu = time.perf_counter() - t
                t = time.perf_counter()
                plan = scorer.plan_from_prepared(indices, prepared, selector=selector, k=k)
                planning = time.perf_counter() - t
                t = time.perf_counter()
                frames = scorer.prepare_dense(path, plan)
                dense_cpu = time.perf_counter() - t
                t = time.perf_counter()
                raw = scorer.score_dense(plan, frames)
                dense_total = time.perf_counter() - t
                t = time.perf_counter()
                windows = raw["windows"]
                gs = percentile(
                    [w["global_spatial_raw"] for w in windows],
                    reference.global_spatial.calibration_raw,
                )
                gt = percentile(
                    [w["global_temporal_raw"] for w in windows],
                    reference.global_temporal.calibration_raw,
                    allow_positive_infinity=True,
                )
                q = window_mean([w["local_raw"] for w in windows])
                score = 0.5 * window_mean(0.5 * gs + 0.5 * gt) + 0.5 * float(
                    percentile([q], cdfs[len(windows)])[0]
                )
                cdf_seconds = time.perf_counter() - t
                rows.append(
                    dict(
                        dataset=domain,
                        video_id=row.video_id,
                        selector=name,
                        repeat=repeat,
                        duration_seconds=row.duration_seconds,
                        total_seconds=time.perf_counter() - start,
                        coarse_decode_prepare_seconds=coarse_cpu,
                        selection_seconds=planning - clock["global_forward"],
                        coarse_forward_seconds=clock["global_forward"],
                        dense_decode_prepare_seconds=dense_cpu,
                        dense_forward_seconds=clock["dense_forward"],
                        score_seconds=dense_total - clock["dense_forward"],
                        cdf_seconds=cdf_seconds,
                        coarse_frames=raw["coarse_frames"],
                        dense_unique_frames=raw["dense_unique_frames"],
                        effective_k=len(windows),
                        observed_unique_frames=len(
                            {f for w in windows for f in w["frame_indices"]}
                            | (set(indices[::8]) if selector == "feature_change" else set())
                        ),
                        coarse_computed_frames=8 * ((raw["coarse_frames"] + 7) // 8),
                        dense_computed_frames=8 * ((raw["dense_unique_frames"] + 7) // 8),
                        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                        peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                        rss_mib=psutil.Process().memory_info().rss / 2**20,
                        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / 1024,
                        final_score=score,
                    )
                )
                del frames, prepared, plan, raw
                if len(rows) % 16 == 0:
                    paper_json(
                        output / "progress.json",
                        dict(
                            completed=len(rows), total=len(sample) * 8, selector=name, repeat=repeat
                        ),
                    )
    pd.DataFrame(rows).to_csv(output / "timings.csv", index=False)
    paper_json(
        output / "manifest.json",
        dict(
            status="completed",
            dataset=domain,
            config=config,
            reference_sha256=file_digest(reference_path),
            initialization_seconds=initialization,
            warmup_seconds=warmups,
            cache="no feature cache; OS page cache uncontrolled; one untimed warmup per selector, repeat0/1 reported separately",
            memory_note="GPU per-video allocator peak; RSS sampled after each video; process_peak_rss is cumulative process high-water mark including setup/warmup, not isolated per-selector RAM",
            decoder="sequential selected frames with original fallback",
            decoder_sha256=file_digest(root / "src/data/sequential_decode.py"),
            concurrency="single process, batch8, synchronous latency measurement; run without competing GPU tasks",
            files={p.name: file_digest(p) for p in output.iterdir() if p.is_file()},
        ),
    )
