"""统一CLI严格配置、无副作用dry-run及raw到固定配对评价的完整小例。"""

import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_paper_config, validate_paper_config
from reference import ReferenceBundle, SCHEMA, save_bundle, file_digest, local_video_cdfs
from math_utils import StableGaussianParams


def test_config_contract():
    path = ROOT / "configs/paper.yaml"
    with pytest.raises(ValueError):
        load_paper_config(path, ["runtime.deivce=cpu"])
    with pytest.raises(ValueError):
        load_paper_config(path, ["encoder.batch_size=64"])
    config = load_paper_config(path)
    config["unknown"] = {}
    with pytest.raises(ValueError):
        validate_paper_config(config)
    assert (
        load_paper_config(path, ["method.local_enabled=false"])["method"]["local_enabled"] is False
    )


def make_cli_case(tmp_path):
    spec = importlib.util.spec_from_file_location("paper_cli", ROOT / "scripts/run.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    params = StableGaussianParams(np.zeros(2), np.eye(2), np.array([0.0, 2.0]))
    bundle = ReferenceBundle(
        params,
        params,
        params,
        local_video_cdfs([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
        dict(schema=SCHEMA, dataset="test"),
    )
    reference = tmp_path / "test.npz"
    save_bundle(reference, bundle)
    digest = file_digest(reference)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(dict(status="completed", references=[dict(dataset="test", sha256=digest)]))
    )
    config = load_paper_config(ROOT / "configs/paper.yaml")
    config["reference"] = dict(directory=str(tmp_path), manifest=str(manifest))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    windows = pd.DataFrame(
        [
            dict(
                video_id=v,
                dataset="test",
                subset=s,
                source_model=g,
                video_path=v + ".mp4",
                window_id=0,
                reference_sha256=digest,
                global_spatial_raw=x,
                global_temporal_raw=x,
                local_raw=x,
            )
            for v, s, g, x in [("r", "real", "real", 2.0), ("f", "annotated", "g", -1.0)]
        ]
    )
    raw = tmp_path / "windows.csv"
    windows.to_csv(raw, index=False)
    run = tmp_path / "run"
    base = [
        "run.py",
        "--config",
        str(path),
        "replay",
        "--window-scores",
        str(raw),
        "--output",
        str(run),
    ]
    return cli, path, windows, run, base


def test_cli_replay_evaluate_and_dry_run(tmp_path, monkeypatch):
    cli, path, windows, run, base = make_cli_case(tmp_path)
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    cli.main()
    assert not run.exists()
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    assert json.loads((run / "status.json").read_text())["completed_steps"] == ["replay"]
    saved = json.loads((run / "run_manifest.json").read_text())
    assert saved["git_commit"]
    assert (run / saved["code_snapshot"] / "src/reference.py").is_file()
    pairs = tmp_path / "pairs.csv"
    windows[["video_id", "dataset", "subset"]].assign(generator="g").to_csv(pairs, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(path), "evaluate", "--run-dir", str(run), "--pairs", str(pairs)],
    )
    cli.main()
    result = pd.read_csv(run / "generator_metrics.csv")
    assert result.iloc[0].auc == 1.0 and result.iloc[0].real_positive_ap == 1.0
    assert "evaluate" in json.loads((run / "status.json").read_text())["completed_steps"]
    assert (run / "logs/console.log").read_text().count("开始") == 2
    before = (run / "video_scores.csv").read_bytes()
    monkeypatch.setattr(sys, "argv", base + ["--resume"])
    cli.main()
    assert (run / "video_scores.csv").read_bytes() == before


def test_interrupted_replay_resumes_without_recomputing_first_video(tmp_path, monkeypatch):
    import workflow

    cli, path, windows, run, base = make_cli_case(tmp_path)
    original = workflow.replay_scores
    calls = []

    def fail_second(frame, *args, **kwargs):
        vid = frame.video_id.iloc[0]
        calls.append(vid)
        if vid == "f":
            raise RuntimeError("模拟中断")
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(workflow, "replay_scores", fail_second)
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(RuntimeError, match="模拟中断"):
        cli.main()
    assert len(list((run / "checkpoints").glob("*.json"))) == 1
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    calls.clear()

    def observe(frame, *args, **kwargs):
        calls.append(frame.video_id.iloc[0])
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(workflow, "replay_scores", observe)
    monkeypatch.setattr(sys, "argv", base + ["--resume"])
    cli.main()
    assert calls == ["f"]
    assert len(pd.read_csv(run / "video_scores.csv")) == 2
    # 改配置后的恢复必须拒绝，不能拿已有checkpoint覆盖成另一项实验。
    monkeypatch.setattr(
        sys, "argv", base[:3] + ["--set", "method.local_enabled=false"] + base[3:] + ["--resume"]
    )
    with pytest.raises(ValueError, match="配置"):
        cli.main()


def test_components_cli_and_explicit_variant(tmp_path, monkeypatch):
    from artifacts import read_paper_scores

    cli, path, windows, run, base = make_cli_case(tmp_path)
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    pairs = tmp_path / "pairs.csv"
    windows[["video_id", "dataset", "subset"]].assign(generator="g").to_csv(pairs, index=False)
    output = tmp_path / "components"
    command = [
        "run.py",
        "--config",
        str(path),
        "components",
        "--run-dir",
        str(run),
        "--pairs",
        str(pairs),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", command + ["--dry-run"])
    cli.main()
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", command)
    cli.main()
    with pytest.raises(ValueError, match="variant"):
        read_paper_scores(output)
    full, identity = read_paper_scores(output, "full")
    old, _ = read_paper_scores(run)
    np.testing.assert_array_equal(full.final_score, old.final_score)
    assert identity["variant"] == "full" and len(pd.read_csv(output / "generator_metrics.csv")) == 5
    with pytest.raises(ValueError, match="不存在"):
        read_paper_scores(output, "not_a_variant")
    snapshots = json.loads((output / "run_manifest.json").read_text())["input_manifest_snapshots"]
    assert len(snapshots) == 1
    for digest, record in snapshots.items():
        assert file_digest(output / record["path"]) == digest


def test_evaluation_lock_and_interrupted_publication(tmp_path, monkeypatch):
    import fcntl
    import artifacts
    import evaluation.tables

    cli, path, windows, run, base = make_cli_case(tmp_path)
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    pairs = tmp_path / "pairs.csv"
    windows[["video_id", "dataset", "subset"]].assign(generator="g").to_csv(pairs, index=False)
    command = [
        "run.py",
        "--config",
        str(path),
        "evaluate",
        "--run-dir",
        str(run),
        "--pairs",
        str(pairs),
    ]
    with (run / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(sys, "argv", command)
        with pytest.raises(BlockingIOError):
            cli.main()
    assert not (run / "stages/evaluate").exists()
    original = artifacts.publish_stage_file
    calls = []

    def interrupt(source, destination, digest):
        calls.append(destination.name)
        if len(calls) == 2:
            raise RuntimeError("模拟评价发布中断")
        original(source, destination, digest)

    monkeypatch.setattr(artifacts, "publish_stage_file", interrupt)
    with pytest.raises(RuntimeError, match="发布中断"):
        cli.main()
    assert (run / "generator_metrics.csv").exists() and not (run / "dataset_metrics.csv").exists()
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert json.loads((run / "stages/evaluate/plan.json").read_text())["status"] == "prepared"
    before = file_digest(run / "video_scores.csv")
    monkeypatch.setattr(artifacts, "publish_stage_file", original)

    def forbid(*args, **kwargs):
        raise AssertionError("准备好的表不应重新计算")

    monkeypatch.setattr(evaluation.tables, "evaluate_fixed_pairs", forbid)
    monkeypatch.setattr(sys, "argv", command + ["--resume"])
    cli.main()
    assert file_digest(run / "video_scores.csv") == before
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    assert (run / "stages/evaluate/code/src/evaluation/tables.py").exists()
    # 空Macro也有schema，而不是无法再次读取的空文件。
    assert pd.read_csv(run / "macro_metrics.csv").empty
    manifest_digest = file_digest(run / "run_manifest.json")
    cli.main()
    assert file_digest(run / "run_manifest.json") == manifest_digest
    pairs.write_text(pairs.read_text() + "\n")
    with pytest.raises(ValueError, match="身份改变"):
        cli.main()


def test_evaluation_keeps_unowned_existing_results(tmp_path, monkeypatch):
    cli, path, windows, run, base = make_cli_case(tmp_path)
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    pairs = tmp_path / "pairs.csv"
    windows[["video_id", "dataset", "subset"]].assign(generator="g").to_csv(pairs, index=False)
    existing = run / "generator_metrics.csv"
    existing.write_text("user-owned result\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--config", str(path), "evaluate", "--run-dir", str(run), "--pairs", str(pairs)],
    )
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        cli.main()
    assert existing.read_text() == "user-owned result\n"
    assert not (run / "stages/evaluate").exists()


def test_compact_score_reference_requires_declared_schema(tmp_path, monkeypatch):
    from artifacts import read_paper_scores, paper_json

    cli, path, windows, run, base = make_cli_case(tmp_path)
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    manifest = json.loads((run / "run_manifest.json").read_text())
    manifest["completed_steps"] = ["import_reference"]
    manifest.pop("code_snapshot", None)
    paper_json(run / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="schema"):
        read_paper_scores(run)
    manifest["schema"] = "paper_score_reference_v1"
    paper_json(run / "run_manifest.json", manifest)
    scores, _ = read_paper_scores(run)
    assert len(scores) == 2
