"""Audit statistical assumptions behind patch-level Gaussian likelihoods.

The original STALL paper validates the Gaussian/whitened-likelihood assumptions
for global frame features and first-order temporal differences. Alpha-STALLED
extends the same likelihood machinery to patch tokens and same-grid second-order
patch differences. This script performs a representative cache-level diagnostic:

1. component-wise AD and D'Agostino-Pearson normality on whitened coordinates;
2. whitened covariance closeness to identity;
3. direction cosine distribution after whitening;
4. whether one shared (mu, W) is tolerable across patch/region positions.

It does not extract features or rebuild calibration parameters. It reads existing
patch cache and patch parameter files.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from create_patch_params import iter_real_patch_cache  # noqa: E402
from patch_matching import patch_temporal_delta, pool_patch_regions  # noqa: E402
from stall import whitening_transform as apply_whitening  # noqa: E402
from whitening_transform import WhiteningTransform  # noqa: E402


@dataclass
class Representation:
    name: str
    rows: np.ndarray
    positions: np.ndarray
    whitening_source: str
    mu: np.ndarray
    W: np.ndarray


def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _sample_rows(x: np.ndarray, max_rows: int, rng: np.random.RandomState) -> np.ndarray:
    if len(x) <= max_rows:
        return x
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx]


def _row_sample_with_positions(
    rows: np.ndarray,
    positions: np.ndarray,
    max_rows: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) <= max_rows:
        return rows, positions
    idx = rng.choice(len(rows), size=max_rows, replace=False)
    return rows[idx], positions[idx]


def _fit_diag_whitening(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wt = WhiteningTransform(data=rows.astype(np.float32))
    return wt.mean_.cpu().numpy(), wt.whitening_matrix_.cpu().numpy()


def _load_patch_params(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def collect_representations(
    csv_path: Path,
    patch_cache: Path,
    patch_params: Path,
    max_real_videos: int,
    max_rows_per_repr: int,
    duration: int,
    compact: bool,
    patch_region_size: int,
    seed: int,
) -> list[Representation]:
    rng = _rng(seed)
    params = _load_patch_params(patch_params)
    mu_spat = params["mu_patch_spat"].astype(np.float32)
    W_spat = params["W_patch_spat"].astype(np.float32)
    mu_temp = params["mu_patch_temp"].astype(np.float32)
    W_temp = params["W_patch_temp"].astype(np.float32)

    raw_rows, raw_pos = [], []
    region_rows, region_pos = [], []
    second_rows, second_pos = [], []
    n_videos = 0
    grid_size_seen = None

    for n_videos, (_, payload) in enumerate(
        iter_real_patch_cache(
            str(csv_path),
            str(patch_cache),
            duration_sec=duration,
            compact=compact,
            max_real_videos=max_real_videos,
        ),
        start=1,
    ):
        patch = payload["patch"].numpy().astype(np.float32)
        grid_size = tuple(int(x) for x in payload["grid_size"])
        if grid_size_seen is None:
            grid_size_seen = grid_size
        elif grid_size_seen != grid_size:
            raise ValueError(f"inconsistent grid size: {grid_size_seen} vs {grid_size}")

        # Raw patch token rows: [T, P, D] -> [T*P, D], positions repeated by time.
        t, p, d = patch.shape
        rows = patch.reshape(t * p, d)
        pos = np.tile(np.arange(p, dtype=np.int32), t)
        rows, pos = _row_sample_with_positions(rows, pos, max(1, max_rows_per_repr // max_real_videos), rng)
        raw_rows.append(rows)
        raw_pos.append(pos)

        pooled, pooled_grid = pool_patch_regions(patch, grid_size, region_size=patch_region_size)
        rt, rp, rd = pooled.shape
        rrows = pooled.reshape(rt * rp, rd)
        rpos = np.tile(np.arange(rp, dtype=np.int32), rt)
        rrows, rpos = _row_sample_with_positions(rrows, rpos, max(1, max_rows_per_repr // max_real_videos), rng)
        region_rows.append(rrows)
        region_pos.append(rpos)

        second = patch_temporal_delta(
            patch,
            grid_size=grid_size,
            mode="same_grid_second_order",
            region_size=patch_region_size,
        )
        st, sp, sd = second.shape
        srows = second.reshape(st * sp, sd)
        spos = np.tile(np.arange(sp, dtype=np.int32), st)
        srows, spos = _row_sample_with_positions(srows, spos, max(1, max_rows_per_repr // max_real_videos), rng)
        second_rows.append(srows)
        second_pos.append(spos)

    if n_videos == 0:
        raise ValueError("No real patch cache rows were found.")

    raw = np.concatenate(raw_rows, axis=0).astype(np.float32)
    raw_p = np.concatenate(raw_pos, axis=0)
    region = np.concatenate(region_rows, axis=0).astype(np.float32)
    region_p = np.concatenate(region_pos, axis=0)
    second = np.concatenate(second_rows, axis=0).astype(np.float32)
    second_p = np.concatenate(second_pos, axis=0)

    raw, raw_p = _row_sample_with_positions(raw, raw_p, max_rows_per_repr, rng)
    region, region_p = _row_sample_with_positions(region, region_p, max_rows_per_repr, rng)
    second, second_p = _row_sample_with_positions(second, second_p, max_rows_per_repr, rng)

    # Region-pooled tokens are an intermediate representation, not directly
    # scored by the release model. Fit a diagnostic whitening on half the sample
    # and evaluate on the held-out half.
    fit_n = max(128, len(region) // 2)
    perm = rng.permutation(len(region))
    fit_idx = perm[:fit_n]
    eval_idx = perm[fit_n:]
    mu_region, W_region = _fit_diag_whitening(region[fit_idx])
    region_eval = region[eval_idx]
    region_pos_eval = region_p[eval_idx]
    if len(region_eval) < 128:
        region_eval = region
        region_pos_eval = region_p

    return [
        Representation(
            name="patch_token",
            rows=raw,
            positions=raw_p,
            whitening_source="release_muW_patch_spatial",
            mu=mu_spat,
            W=W_spat,
        ),
        Representation(
            name=f"region{patch_region_size}_pooled_token",
            rows=region_eval.astype(np.float32),
            positions=region_pos_eval,
            whitening_source="diagnostic_half_split_muW",
            mu=mu_region.astype(np.float32),
            W=W_region.astype(np.float32),
        ),
        Representation(
            name=f"region{patch_region_size}_second_order_diff",
            rows=second,
            positions=second_p,
            whitening_source="release_muW_patch_temporal",
            mu=mu_temp,
            W=W_temp,
        ),
    ]


def whiten(rows: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return apply_whitening(rows.astype(np.float32), mu.astype(np.float32), W.astype(np.float32)).astype(np.float32)


def covariance_summary(z: np.ndarray) -> dict[str, float]:
    cov = np.cov(z, rowvar=False)
    dim = cov.shape[0]
    diag = np.diag(cov)
    off = cov.copy()
    np.fill_diagonal(off, 0.0)
    eye = np.eye(dim)
    return {
        "whitened_dim": int(dim),
        "n_rows": int(len(z)),
        "coord_mean_abs": float(np.mean(np.abs(z.mean(axis=0)))),
        "diag_mean": float(np.mean(diag)),
        "diag_std": float(np.std(diag)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "offdiag_rms": float(np.sqrt(np.sum(off * off) / max(1, dim * (dim - 1)))),
        "offdiag_abs_p95": float(np.percentile(np.abs(off[~np.eye(dim, dtype=bool)]), 95)),
        "cov_fro_rel": float(np.linalg.norm(cov - eye, ord="fro") / math.sqrt(dim)),
    }


def normality_summary(z: np.ndarray, max_dims: int, max_rows: int, seed: int) -> dict[str, float]:
    rng = _rng(seed)
    rows = _sample_rows(z, min(max_rows, len(z)), rng)
    dim = rows.shape[1]
    dims = np.arange(dim)
    if dim > max_dims:
        dims = rng.choice(dim, size=max_dims, replace=False)

    dp_pvals = []
    ad_pass_5 = []
    ad_stats = []
    skew_vals = []
    kurt_vals = []
    for j in dims:
        x = rows[:, j]
        try:
            dp_pvals.append(float(stats.normaltest(x).pvalue))
        except Exception:
            dp_pvals.append(np.nan)
        ad = stats.anderson(x, dist="norm")
        crit_5 = ad.critical_values[list(ad.significance_level).index(5.0)]
        ad_stats.append(float(ad.statistic))
        ad_pass_5.append(float(ad.statistic < crit_5))
        skew_vals.append(float(stats.skew(x, bias=False)))
        kurt_vals.append(float(stats.kurtosis(x, fisher=True, bias=False)))

    dp = np.array(dp_pvals, dtype=np.float64)
    return {
        "normality_rows": int(len(rows)),
        "normality_dims": int(len(dims)),
        "dp_pass_p_gt_0p01": float(np.nanmean(dp > 0.01)),
        "dp_pass_p_gt_0p05": float(np.nanmean(dp > 0.05)),
        "dp_median_p": float(np.nanmedian(dp)),
        "ad_pass_5pct": float(np.mean(ad_pass_5)),
        "ad_median_stat": float(np.median(ad_stats)),
        "abs_skew_median": float(np.median(np.abs(skew_vals))),
        "abs_excess_kurtosis_median": float(np.median(np.abs(kurt_vals))),
    }


def direction_summary(z: np.ndarray, n_pairs: int, seed: int) -> dict[str, float]:
    rng = _rng(seed)
    if len(z) < 2:
        raise ValueError("Need at least two rows for direction cosine diagnostic.")
    dim = z.shape[1]
    idx_a = rng.randint(0, len(z), size=n_pairs)
    idx_b = rng.randint(0, len(z), size=n_pairs)
    a = z[idx_a].astype(np.float64)
    b = z[idx_b].astype(np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    cos = np.sum(a * b, axis=1)
    expected_std = 1.0 / math.sqrt(dim)
    ks = stats.kstest(cos, "norm", args=(0.0, expected_std))
    return {
        "direction_pairs": int(n_pairs),
        "cos_mean": float(np.mean(cos)),
        "cos_std": float(np.std(cos)),
        "cos_expected_std": float(expected_std),
        "cos_abs_mean": float(np.mean(np.abs(cos))),
        "cos_abs_expected_mean": float(math.sqrt(2.0 / (math.pi * dim))),
        "cos_ks_stat_vs_spherical_normal": float(ks.statistic),
        "cos_ks_p_vs_spherical_normal": float(ks.pvalue),
    }


def position_summary(z: np.ndarray, positions: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    ll = -0.5 * np.sum(z.astype(np.float64) ** 2, axis=1)
    norm2_per_dim = np.sum(z.astype(np.float64) ** 2, axis=1) / z.shape[1]
    rows = []
    for pos in sorted(np.unique(positions)):
        mask = positions == pos
        if mask.sum() < 8:
            continue
        rows.append(
            {
                "position": int(pos),
                "n_rows": int(mask.sum()),
                "mean_ll": float(ll[mask].mean()),
                "std_ll": float(ll[mask].std()),
                "mean_norm2_per_dim": float(norm2_per_dim[mask].mean()),
                "mean_abs_coord_mean": float(np.mean(np.abs(z[mask].mean(axis=0)))),
            }
        )
    df = pd.DataFrame(rows)
    global_ll_mean = float(ll.mean())
    out = {
        "n_positions": int(len(df)),
        "position_mean_ll_std": float(df["mean_ll"].std()) if len(df) else np.nan,
        "position_mean_ll_range": float(df["mean_ll"].max() - df["mean_ll"].min()) if len(df) else np.nan,
        "position_norm2_per_dim_std": float(df["mean_norm2_per_dim"].std()) if len(df) else np.nan,
        "position_max_abs_ll_shift": float(np.max(np.abs(df["mean_ll"] - global_ll_mean))) if len(df) else np.nan,
    }
    return out, df


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["diag_mean"], width=0.36, label="diag mean")
    ax.bar(x + 0.18, summary["offdiag_rms"], width=0.36, label="offdiag RMS")
    ax.axhline(1.0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(x, summary["display"], rotation=20, ha="right")
    ax.set_title("Whitened covariance")
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(x - 0.18, summary["ad_pass_5pct"], width=0.36, label="AD pass@5%")
    ax.bar(x + 0.18, summary["dp_pass_p_gt_0p01"], width=0.36, label="DP p>0.01")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, summary["display"], rotation=20, ha="right")
    ax.set_title("Coordinate normality")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(x - 0.18, summary["cos_std"], width=0.36, label="observed")
    ax.bar(x + 0.18, summary["cos_expected_std"], width=0.36, label="sphere expected")
    ax.set_xticks(x, summary["display"], rotation=20, ha="right")
    ax.set_title("Direction cosine std")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(x, summary["position_norm2_per_dim_std"], color="#E6862A")
    ax.set_xticks(x, summary["display"], rotation=20, ha="right")
    ax.set_title("Position-sharing residual")
    ax.set_ylabel("std of position mean norm²/dim")
    ax.grid(axis="y", alpha=0.25)

    for ext in ("pdf", "png", "svg"):
        path = out_dir / f"patch_likelihood_assumption_audit.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=260)
        else:
            fig.savefig(path)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, out_dir: Path, max_real_videos: int) -> None:
    lines = [
        "# Patch likelihood statistical assumption audit",
        "",
        f"Dataset: ComGenVid real videos; sampled real videos: {max_real_videos}.",
        "",
        "This diagnostic checks whether the Gaussian likelihood machinery inherited from STALL is at least approximately reasonable for patch-level representations.",
        "It is not a proof of exact multivariate normality. With thousands of samples and high-dimensional neural features, strict normality tests are expected to be sensitive to small deviations.",
        "",
        "## Summary",
        "",
        "| representation | whitening | rows | dim | diag mean | offdiag RMS | AD pass@5% | DP p>0.01 | cosine std / expected | position norm² std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        lines.append(
            f"| {r.representation} | {r.whitening_source} | {int(r.n_rows)} | {int(r.whitened_dim)} | "
            f"{r.diag_mean:.3f} | {r.offdiag_rms:.3f} | {r.ad_pass_5pct:.3f} | {r.dp_pass_p_gt_0p01:.3f} | "
            f"{r.cos_std:.4f}/{r.cos_expected_std:.4f} | {r.position_norm2_per_dim_std:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation for the manuscript",
            "",
            "- The covariance and direction-cosine diagnostics are the most relevant checks for using a whitened Gaussian score. They test whether whitening approximately sphericalizes the selected representation.",
            "- AD and D'Agostino-Pearson tests are reported as stress diagnostics rather than pass/fail proof. Low pass rates should be written as a modeling boundary, not as a failure of the detector.",
            "- Position-sharing is assessed by the spread of per-position whitened norm² and log-likelihood means. A small spread supports using one shared `(mu, W)`; a large spread would motivate per-position calibration.",
            "",
            "## Files",
            "",
            "- `patch_likelihood_assumption_summary.csv`",
            "- `patch_likelihood_position_summary.csv`",
            "- `patch_likelihood_position_detail.csv`",
            "- `patch_likelihood_assumption_audit.pdf/.png/.svg`",
        ]
    )
    (out_dir / "patch_likelihood_assumption_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(summary: pd.DataFrame, paper_table_dir: Path) -> None:
    display_map = {
        "patch_token": "Patch token",
        "region3_pooled_token": "Region-pooled token",
        "region3_second_order_diff": "Second-order diff",
    }
    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{Patch 似然统计假设诊断。实验使用 ComGenVid 真实视频 patch cache 抽样；AD 和 D'Agostino--Pearson (DP) 为坐标级正态性压力测试，covariance 和 direction cosine 用于检验白化后近似球形性。}",
        r"  \label{tab:patch_likelihood_assumption_audit}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lccccccc}",
        r"    \toprule",
        r"    表示 & 行数 & 白化维度 & Cov diag mean & Offdiag RMS & AD pass@5\% & DP $p>0.01$ & Cos std / expected \\",
        r"    \midrule",
    ]
    for r in summary.itertuples(index=False):
        name = display_map.get(r.representation, r.representation)
        lines.append(
            "    "
            + " & ".join(
                [
                    name,
                    f"{int(r.n_rows)}",
                    f"{int(r.whitened_dim)}",
                    f"{r.diag_mean:.3f}",
                    f"{r.offdiag_rms:.3f}",
                    f"{r.ad_pass_5pct:.3f}",
                    f"{r.dp_pass_p_gt_0p01:.3f}",
                    f"{r.cos_std:.4f}/{r.cos_expected_std:.4f}",
                ]
            )
            + r" \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]
    (paper_table_dir / "patch_likelihood_assumption_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "cache/indexes/comgenvid.csv")
    parser.add_argument("--patch-cache", type=Path, default=ROOT / "cache/patch_embeddings/comgenvid")
    parser.add_argument(
        "--patch-params",
        type=Path,
        default=ROOT / "precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/journal_experiments/patch_likelihood_assumption_audit",
    )
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper/ieee_alpha_stalled")
    parser.add_argument("--max-real-videos", type=int, default=256)
    parser.add_argument("--max-rows-per-repr", type=int, default=50000)
    parser.add_argument("--normality-dims", type=int, default=128)
    parser.add_argument("--normality-rows", type=int, default=5000)
    parser.add_argument("--direction-pairs", type=int, default=20000)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--patch-region-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.paper_dir / "figures/results").mkdir(parents=True, exist_ok=True)

    reps = collect_representations(
        csv_path=args.csv,
        patch_cache=args.patch_cache,
        patch_params=args.patch_params,
        max_real_videos=args.max_real_videos,
        max_rows_per_repr=args.max_rows_per_repr,
        duration=args.duration,
        compact=True,
        patch_region_size=args.patch_region_size,
        seed=args.seed,
    )

    summary_rows = []
    position_frames = []
    position_summary_rows = []
    for i, rep in enumerate(reps):
        print(f"auditing {rep.name}: rows={rep.rows.shape}", flush=True)
        z = whiten(rep.rows, rep.mu, rep.W)
        cov = covariance_summary(z)
        norm = normality_summary(z, args.normality_dims, args.normality_rows, args.seed + i)
        direction = direction_summary(z, args.direction_pairs, args.seed + 100 + i)
        pos_sum, pos_df = position_summary(z, rep.positions)
        pos_df.insert(0, "representation", rep.name)
        position_frames.append(pos_df)
        position_summary_rows.append({"representation": rep.name, **pos_sum})
        summary_rows.append(
            {
                "representation": rep.name,
                "display": rep.name.replace("_", "\n"),
                "whitening_source": rep.whitening_source,
                **cov,
                **norm,
                **direction,
                **pos_sum,
            }
        )

    summary = pd.DataFrame(summary_rows)
    pos_summary = pd.DataFrame(position_summary_rows)
    pos_detail = pd.concat(position_frames, ignore_index=True)
    summary.to_csv(args.out_dir / "patch_likelihood_assumption_summary.csv", index=False)
    pos_summary.to_csv(args.out_dir / "patch_likelihood_position_summary.csv", index=False)
    pos_detail.to_csv(args.out_dir / "patch_likelihood_position_detail.csv", index=False)

    plot_summary(summary, args.out_dir)
    write_markdown(summary, args.out_dir, args.max_real_videos)
    write_latex_table(summary, args.paper_dir / "tables")

    # Copy figure assets into the manuscript folder.
    for ext in ("pdf", "png", "svg"):
        src = args.out_dir / f"patch_likelihood_assumption_audit.{ext}"
        dst = args.paper_dir / "figures/results" / f"patch_likelihood_assumption_audit.{ext}"
        dst.write_bytes(src.read_bytes())

    print(summary.to_string(index=False))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
