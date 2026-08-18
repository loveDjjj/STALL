#!/usr/bin/env python3
"""Verify that project entry documents agree with the locked U0 release."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.cache_contract import CONTRACT_FILENAME
from alpha_stalled.cache_inventory import render_cache_inventory, validate_cache_inventory
from alpha_stalled.config_registry import (
    discover_config_assets,
    read_config_registry,
    validate_config_registry,
)
from alpha_stalled.data_catalog import render_data_catalog, validate_data_catalog
from alpha_stalled.experiment_registry import read_registry, validate_registry
from alpha_stalled.environment_lock import validate_environment_files
from alpha_stalled.parameter_assets import (
    read_parameter_assets,
    render_parameter_asset_report,
    validate_parameter_assets,
)
from alpha_stalled.run_manifest import validate_run_manifest
from alpha_stalled.release_index import validate_release_indexes
from alpha_stalled.tool_dependencies import (
    dependency_inventory,
    read_tool_dependency_policy,
    render_dependency_report,
)


CONFIG = ROOT / "configs/alpha_stalled_u0_locked.yaml"
RELEASE = ROOT / "release/u0"
REGISTRY = ROOT / "reports/u0_experiment_registry.csv"
RUN_MANIFEST = (
    ROOT / "results/research_summary/run_manifests/alpha_stalled_u0_locked.json"
)
CACHE_INVENTORY_SPEC = ROOT / "configs/cache_inventory.yaml"
CACHE_INVENTORY = ROOT / "results/research_summary/cache_inventory.json"
CACHE_INVENTORY_REPORT = ROOT / "reports/cache_inventory.md"
CONFIG_REGISTRY = ROOT / "configs/config_registry.yaml"
DATA_CATALOG_SPEC = ROOT / "configs/data_catalog.yaml"
DATA_CATALOG = ROOT / "results/research_summary/data_catalog.json"
DATA_CATALOG_REPORT = ROOT / "reports/data_catalog.md"
PARAMETER_ASSET_SPEC = ROOT / "configs/parameter_assets.yaml"
PARAMETER_ASSET_REPORT = ROOT / "reports/parameter_asset_inventory.md"
TOOL_DEPENDENCY_SPEC = ROOT / "configs/tool_dependencies.yaml"
TOOL_DEPENDENCY_INVENTORY = (
    ROOT / "results/research_summary/tool_dependency_inventory.json"
)
TOOL_DEPENDENCY_REPORT = ROOT / "reports/tool_dependency_inventory.md"
FEATURE_CACHE_CONTRACT_REPORT = ROOT / "reports/feature_cache_contract.md"
LOCAL_BRANCH_BOUNDARY_REPORT = ROOT / "reports/local_branch_code_boundary.md"
GLOBAL_BRANCH_BOUNDARY_REPORT = ROOT / "reports/global_branch_code_boundary.md"
BACKBONE_BOUNDARY_REPORT = ROOT / "reports/backbone_code_boundary.md"
VIDEO_IO_BOUNDARY_REPORT = ROOT / "reports/video_io_code_boundary.md"
U0_PROTOCOL_BOUNDARY_REPORT = ROOT / "reports/u0_protocol_code_boundary.md"
SCORE_CSV_BOUNDARY_REPORT = ROOT / "reports/score_csv_code_boundary.md"
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "FILE_STRUCTURE.md",
    ROOT / "docs/CURRENT_PROTOCOL.md",
    ROOT / "docs/RESULT_STATUS.md",
    ROOT / "docs/PROJECT_ORGANIZATION.md",
    ROOT / "docs/restructure/README.md",
    ROOT / "docs/restructure/asset_manifest.md",
    ROOT / "docs/restructure/file_inventory.md",
    ROOT / "docs/restructure/release_scope.md",
    ROOT / "docs/restructure/release_staging_manifest.md",
    ROOT / "docs/restructure/reproducibility_audit.md",
    ROOT / "configs/README.md",
    ROOT / "configs/run_manifests/README.md",
    ROOT / "release/README.md",
    ROOT / "release/u0/README.md",
    ROOT / "release/u0_external_genvidbench/README.md",
    ROOT / "reports/README.md",
    ROOT / "scripts/README.md",
    ROOT / "scripts/reproduce/README.md",
    ROOT / "scripts/ablations/README.md",
    ROOT / "scripts/experiments/README.md",
    ROOT / "results/README.md",
    ROOT / "results/research_summary/README.md",
    ROOT / "research_archive/docs/pre_u0_release/README.md",
    CACHE_INVENTORY_REPORT,
    DATA_CATALOG_REPORT,
    PARAMETER_ASSET_REPORT,
    TOOL_DEPENDENCY_REPORT,
    FEATURE_CACHE_CONTRACT_REPORT,
    LOCAL_BRANCH_BOUNDARY_REPORT,
    GLOBAL_BRANCH_BOUNDARY_REPORT,
    BACKBONE_BOUNDARY_REPORT,
    VIDEO_IO_BOUNDARY_REPORT,
    U0_PROTOCOL_BOUNDARY_REPORT,
    SCORE_CSV_BOUNDARY_REPORT,
)
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_fragments(path: Path, fragments: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    checks = []
    for fragment in fragments:
        require(fragment in text, f"{path.relative_to(ROOT)} missing: {fragment!r}")
        checks.append(f"{path.relative_to(ROOT)} contains {fragment!r}")
    return checks


def verify_markdown_links(path: Path) -> list[str]:
    checks = []
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        require(
            resolved.exists(),
            f"{path.relative_to(ROOT)} has broken local link: {raw_target}",
        )
        checks.append(f"{path.relative_to(ROOT)} link exists: {raw_target}")
    return checks


def verify(root: Path = ROOT) -> list[str]:
    require(root.resolve() == ROOT.resolve(), "custom repository roots are not supported")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads((RELEASE / "reproduction_metadata.json").read_text())
    validation = json.loads((RELEASE / "validation.json").read_text())
    calibration = json.loads((RELEASE / "calibration_manifest.json").read_text())
    evaluation = json.loads((RELEASE / "evaluation_manifest.json").read_text())
    registry_rows, registry_columns = read_registry(REGISTRY)
    registry_summary = validate_registry(
        registry_rows, registry_columns, repository_root=ROOT
    )
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    run_summary = validate_run_manifest(
        run_manifest, repository_root=ROOT, verify_hashes=False
    )
    cache_spec = yaml.safe_load(CACHE_INVENTORY_SPEC.read_text(encoding="utf-8"))
    cache_inventory = json.loads(CACHE_INVENTORY.read_text(encoding="utf-8"))
    cache_summary = validate_cache_inventory(
        cache_inventory,
        repository_root=ROOT,
        verify_layout=False,
    )
    config_registry = read_config_registry(CONFIG_REGISTRY)
    config_registry_summary = validate_config_registry(config_registry, ROOT)
    cache_declaration = copy.deepcopy(cache_inventory)
    cache_declaration.pop("summary", None)
    for group in cache_declaration["groups"]:
        group.pop("stats", None)
    require(cache_declaration == cache_spec, "cache inventory/spec declarations differ")
    require(
        CACHE_INVENTORY_REPORT.read_text(encoding="utf-8")
        == render_cache_inventory(cache_inventory),
        "cache inventory Markdown is stale",
    )
    data_spec = yaml.safe_load(DATA_CATALOG_SPEC.read_text(encoding="utf-8"))
    data_catalog = json.loads(DATA_CATALOG.read_text(encoding="utf-8"))
    data_summary = validate_data_catalog(
        data_catalog,
        repository_root=ROOT,
        verify_inputs=False,
    )
    data_declaration = copy.deepcopy(data_catalog)
    data_declaration["schema_version"] = data_spec["schema_version"]
    data_declaration.pop("summary", None)
    data_declaration["release"].pop("calibration_manifest_sha256", None)
    data_declaration["release"].pop("evaluation_manifest_sha256", None)
    for dataset in data_declaration["datasets"]:
        dataset.pop("stats", None)
        dataset.pop("sources", None)
    require(data_declaration == data_spec, "data catalog/spec declarations differ")
    require(
        DATA_CATALOG_REPORT.read_text(encoding="utf-8")
        == render_data_catalog(data_catalog),
        "data catalog Markdown is stale",
    )
    parameter_assets = read_parameter_assets(PARAMETER_ASSET_SPEC)
    parameter_summary = validate_parameter_assets(parameter_assets, ROOT)
    require(
        PARAMETER_ASSET_REPORT.read_text(encoding="utf-8")
        == render_parameter_asset_report(parameter_assets, parameter_summary),
        "parameter asset Markdown is stale",
    )
    tool_dependency_policy = read_tool_dependency_policy(TOOL_DEPENDENCY_SPEC)
    expected_tool_inventory = dependency_inventory(ROOT / "tools", tool_dependency_policy)
    tool_inventory = json.loads(TOOL_DEPENDENCY_INVENTORY.read_text(encoding="utf-8"))
    require(
        tool_inventory == expected_tool_inventory,
        "tool dependency inventory does not match current AST/policy",
    )
    require(
        TOOL_DEPENDENCY_REPORT.read_text(encoding="utf-8")
        == render_dependency_report(tool_inventory),
        "tool dependency Markdown is stale",
    )
    environment_summary = validate_environment_files(ROOT)

    protocol_id = str(config["release"]["protocol_version"])
    macro_auc = float(config["release"]["stable_macro_auc"])
    macro_ap = float(config["release"]["stable_macro_real_positive_ap"])
    expected_evaluation = config["evaluation"]["expected_evaluation_counts"]
    expected_calibration = config["evaluation"]["expected_calibration_real_counts"]

    checks = []
    require(protocol_id == "u0_locked_v1", f"unexpected protocol ID: {protocol_id}")
    checks.append("locked protocol ID is u0_locked_v1")
    require(
        registry_summary.main_experiment_id == "alpha_stalled_u0_locked",
        "registry locked main experiment mismatch",
    )
    require(
        registry_summary.status_counts.get("coverage_extension", 0) >= 1,
        "registry is missing the full-coverage extension",
    )
    locked_registry_row = {
        row["experiment_id"]: row for row in registry_rows
    }[registry_summary.main_experiment_id]
    require(
        locked_registry_row["protocol_id"] == protocol_id,
        "registry/config protocol ID mismatch",
    )
    require(
        float(locked_registry_row["macro_auc"]) == macro_auc
        and float(locked_registry_row["macro_ap"]) == macro_ap,
        "registry/config locked metrics mismatch",
    )
    checks.append(
        f"experiment registry is valid with {registry_summary.row_count} rows"
    )
    require(
        run_summary.experiment_id == registry_summary.main_experiment_id,
        "locked run manifest/registry experiment ID mismatch",
    )
    require(run_summary.artifact_count == 28, "unexpected locked run artifact count")
    checks.append("locked run manifest is valid with 28 artifacts")
    require(cache_summary.group_count == 13, "unexpected cache inventory group count")
    require(
        cache_inventory["summary"]["unregistered_file_count"] == 0,
        "cache inventory snapshot contains unregistered files",
    )
    checks.append("cache inventory declarations and generated report agree")
    require(config_registry_summary.asset_count == 9, "unexpected config asset count")
    require(
        config_registry_summary.protocol_config_count == 1,
        "unexpected protocol config count",
    )
    require(
        config_registry_summary.current_path
        == "configs/alpha_stalled_u0_locked.yaml",
        "unexpected current config authority",
    )
    checks.append(
        "config registry covers 9 assets with one locked current authority"
    )
    require(data_summary.dataset_count == 3, "unexpected canonical dataset count")
    require(data_summary.canonical_video_count == 60949, "unexpected canonical video count")
    require(data_summary.release_video_count == 22021, "unexpected catalog release count")
    require(data_summary.missing_file_count == 0, "data catalog reports missing videos")
    checks.append("data catalog declarations, identities, and generated report agree")
    require(parameter_summary.governed_asset_count == 5, "unexpected parameter asset count")
    require(parameter_summary.current_release_count == 4, "unexpected current parameter count")
    require(
        parameter_summary.external_confirmation_count == 1,
        "unexpected external parameter count",
    )
    require(
        parameter_summary.historical_frozen_count == 0,
        "unexpected historical parameter count",
    )
    checks.append("parameter identities, NPZ contracts, and generated report agree")
    require(tool_inventory["tool_module_count"] == 126, "unexpected Python tool count")
    require(tool_inventory["edge_count"] == 8, "unexpected internal tool edge count")
    require(tool_inventory["retained_family_count"] == 4, "unexpected retained family count")
    require(
        tool_inventory["lifecycle_counts"]
        == {
            "compatibility": 16,
            "formal_release": 10,
            "governance": 16,
            "historical_frozen": 38,
            "paper_evidence": 45,
            "research_utility": 1,
        },
        "unexpected tool lifecycle counts",
    )
    require(len(tool_inventory["modules"]) == 126, "incomplete tool lifecycle inventory")
    require(
        tool_inventory["maintenance_decision_counts"]
        == {
            "compatibility_wrapper": 14,
            "retain_referenced": 3,
        },
        "unexpected tool maintenance decision counts",
    )
    require(
        len(tool_inventory["maintenance_decisions"]) == 17,
        "incomplete tool maintenance decision inventory",
    )
    require(
        tool_inventory["retained_family_status_counts"]
        == {"frozen_historical": 2, "frozen_supplement": 2},
        "unexpected retained family status counts",
    )
    require(
        len(tool_inventory["isolated_entrypoints"]) == 50,
        "unexpected isolated tool entrypoint count",
    )
    checks.append("tool dependency policy, AST graph, and generated report agree")
    require(environment_summary.direct_pip_count == 16, "unexpected direct pip count")
    require(environment_summary.locked_conda_count == 148, "unexpected locked conda count")
    require(environment_summary.locked_pip_count == 94, "unexpected locked pip count")
    require(environment_summary.external_import_count == 15, "unexpected external import count")
    checks.append("direct and exact environment declarations cover runtime imports")
    feature_groups = [
        group
        for group in cache_inventory["groups"]
        if group["family"] in {"global_embeddings", "patch_embeddings"}
    ]
    require(len(feature_groups) == 10, "unexpected feature cache root count")
    contracted_roots = [
        group["path"]
        for group in feature_groups
        if (ROOT / group["path"] / CONTRACT_FILENAME).is_file()
    ]
    require(
        not contracted_roots,
        f"feature cache report is stale; strict roots now exist: {contracted_roots}",
    )
    checks.append("all ten inventoried feature cache roots remain explicitly legacy")
    require(metadata["macro_auc"] == macro_auc, "config/metadata Macro AUC mismatch")
    require(
        metadata["macro_real_positive_ap"] == macro_ap,
        "config/metadata Macro AP mismatch",
    )
    require(validation["passed"], "release validation is not passed")
    require(validation["macro_auc"] == macro_auc, "validation Macro AUC mismatch")
    require(
        validation["macro_real_positive_ap"] == macro_ap,
        "validation Macro AP mismatch",
    )
    checks.append("config, metadata, and validation metrics agree")

    require(calibration["video_count"] == expected_calibration["total"], "calibration count mismatch")
    require(evaluation["video_count"] == expected_evaluation["total"], "evaluation count mismatch")
    require(
        calibration["dataset_counts"]
        == {key: expected_calibration[key] for key in calibration["dataset_counts"]},
        "calibration dataset counts mismatch",
    )
    require(
        evaluation["dataset_counts"]
        == {key: expected_evaluation[key] for key in evaluation["dataset_counts"]},
        "evaluation dataset counts mismatch",
    )
    checks.append("release manifest counts agree with the locked config")

    rounded_auc = f"{macro_auc:.6f}"
    rounded_ap = f"{macro_ap:.6f}"
    checks.extend(
        require_fragments(
            ROOT / "README.md",
            [
                protocol_id,
                "configs/alpha_stalled_u0_locked.yaml",
                "release/u0/final_video_scores.csv",
                rounded_auc,
                rounded_ap,
                "tools/verify_u0_locked_release.py",
                "tools/verify_u0_pipeline_reconstruction.py",
                "tools/verify_experiment_registry.py",
                "tools/verify_run_manifest.py",
                "tools/capture_experiment_run.py",
                "tools/verify_run_capture.py",
                "tools/verify_cache_inventory.py",
                "tools/verify_data_catalog.py",
                "tools/verify_feature_cache_contract.py",
                "tools/verify_tool_dependencies.py",
                "tools/verify_environment_lock.py",
                "environment.lock.yml",
                "configs/cache_inventory.yaml",
                "configs/data_catalog.yaml",
                "configs/tool_dependencies.yaml",
                "reports/feature_cache_contract.md",
                "reports/data_catalog.md",
                "docs/CURRENT_PROTOCOL.md",
                "docs/RESULT_STATUS.md",
                "docs/PROJECT_ORGANIZATION.md",
                "configs/README.md",
                "reports/README.md",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "docs/CURRENT_PROTOCOL.md",
            [
                protocol_id,
                str(macro_auc),
                str(macro_ap),
                "21,421",
                "58,496" if metadata["raw_window_count"] == 58496 else str(metadata["raw_window_count"]),
                "AP_real",
                "AP_fake",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "docs/RESULT_STATUS.md",
            [
                "main_method_locked",
                "invalid_leakage",
                "0.8741/0.8723",
                "0.8737/0.8750",
                "0.9198/0.9211",
                "duration_aware_23source",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "scripts/reproduce/README.md",
            [protocol_id, "Legacy K1 paper_scores", "verify_u0_locked_release.py"],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "docs/PROJECT_ORGANIZATION.md",
            [
                protocol_id,
                "run_manifest.json",
                "src/alpha_stalled/",
                "release_io.py",
                "experiment_registry.py",
                "verify_experiment_registry.py",
                "run_manifest.py",
                "run_capture.py",
                "verify_run_manifest.py",
                "capture_experiment_run.py",
                "verify_run_capture.py",
                "results/runs/<experiment_id>/",
                "Cache key",
                "cache_inventory.yaml",
                "P3_safe_delete_candidate",
                "cache_contract.py",
                "verify_feature_cache_contract.py",
                "feature_cache_contract.md",
                "create_patch_params.py",
                "legacy_uncontracted",
                "alpha_stalled/metrics.py",
                "兼容转发",
                "alpha_stalled/whitening.py",
                "right-inclusive ECDF",
                "alpha_stalled/local_branch.py",
                "local_branch_code_boundary.md",
                "alpha_stalled/global_branch.py",
                "global_branch_code_boundary.md",
                "alpha_stalled/backbone.py",
                "backbone_code_boundary.md",
                "alpha_stalled/video_io.py",
                "video_io_code_boundary.md",
                "alpha_stalled/u0_protocol.py",
                "alpha_stalled/u0_scoring.py",
                "alpha_stalled/u0_analysis.py",
                "u0_protocol_code_boundary.md",
                "verify_u0_pipeline_reconstruction.py",
                "data_catalog.yaml",
                "data_catalog.json",
                "60,949",
                "tool_dependencies.yaml",
                "tool_dependency_inventory.json",
                "verify_tool_dependencies.py",
                "verify_environment_lock.py",
                "environment.lock.yml",
                "8 条",
                "legacy_window_scoring.py",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "configs/README.md",
            [
                protocol_id,
                "u0_locked_v1",
                "当前锁定主方法",
                "multi_window_exclusions.csv",
                "run_manifests/alpha_stalled_u0_locked.yaml",
                "run_manifests/captured_experiment.yaml.template",
                "build_run_manifest.py",
                "capture_experiment_run.py",
                "verify_run_capture.py",
                "cache_inventory.yaml",
                "verify_cache_inventory.py",
                "cache_contract.py",
                "data_catalog.yaml",
                "build_data_catalog.py",
                "verify_data_catalog.py",
                "tool_dependencies.yaml",
                "build_tool_dependency_inventory.py",
                "verify_tool_dependencies.py",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "configs/run_manifests/README.md",
            [
                "captured",
                "reconstructed",
                "provenance_gaps",
                "build_run_manifest.py",
                "verify_run_manifest.py",
                "capture_experiment_run.py",
                "verify_run_capture.py",
                "provenance.capture_path",
                "results/runs/<experiment_id>/run_manifest.json",
                "--skip-hashes",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "reports/README.md",
            [
                "u0_release_reproduction.md",
                "second_order_and_independent_calibration.md",
                "u0_experiment_registry.csv",
                "verify_experiment_registry.py",
                "cache_inventory.md",
                "data_catalog.md",
                "feature_cache_contract.md",
                "local_branch_code_boundary.md",
                "global_branch_code_boundary.md",
                "backbone_code_boundary.md",
                "video_io_code_boundary.md",
                "u0_protocol_code_boundary.md",
                "tool_dependency_inventory.md",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "results/README.md",
            [
                "release/u0/final_video_scores.csv",
                "Legacy K1",
                "verify_u0_locked_release.py",
                "u0_experiment_registry.csv",
                "verify_experiment_registry.py",
                "run_manifests/",
                "cache_inventory.json",
                "data_catalog.json",
                "results/runs/<experiment_id>/",
                "run_capture.json",
                "tool_dependency_inventory.json",
            ],
        )
    )
    checks.extend(
        require_fragments(
            ROOT / "scripts/experiments/README.md",
            [
                "capture_experiment_run.py",
                "results/runs/<experiment_id>/",
                "run_capture.json",
                "--require-clean",
                "verify_run_capture.py",
                "--allow-running",
                "captured_experiment.yaml.template",
                "u0_experiment_registry.csv",
            ],
        )
    )
    require(
        (ROOT / "configs/run_manifests/captured_experiment.yaml.template").is_file(),
        "missing captured experiment manifest template",
    )
    checks.append("captured experiment manifest template exists")
    checks.extend(
        require_fragments(
            ROOT / "results/research_summary/README.md",
            [
                "run_manifests/alpha_stalled_u0_locked.json",
                "build_run_manifest.py",
                "verify_run_manifest.py",
                "reconstructed",
                "cache_inventory.json",
                "verify_cache_inventory.py",
                "data_catalog.json",
                "data_catalog.md",
                "u0_protocol_code_boundary.md",
                "cache_contract.py",
                "feature_cache_contract.md",
                "video_io_code_boundary.md",
                "u0_protocol.py",
                "u0_scoring.py",
                "u0_analysis.py",
                "u0_protocol_code_boundary.md",
                "tool_dependency_inventory.json",
                "verify_tool_dependencies.py",
            ],
        )
    )
    checks.extend(
        require_fragments(
            FEATURE_CACHE_CONTRACT_REPORT,
            [
                "97,900",
                "strict_evidence=false",
                ".alpha_stalled_cache_contract.json",
                ".pt.meta.json",
                "checkpoint_sha256",
                "--verify-cache-hashes",
                "feature_cache_contract_sha256",
                "legacy_uncontracted",
                "Historical research tools that directly call `torch.load`",
                "decode_indexed_frames(..., require_all=True)",
            ],
        )
    )
    checks.extend(
        require_fragments(
            DATA_CATALOG_REPORT,
            [
                "data_catalog_20260814",
                "Canonical video identities: 60,949",
                "Locked U0 identities: 22,021",
                "Canonical identities outside locked U0: 38,928",
                "Missing source files on this workspace: 0",
                "Hotshot-XL",
                "MoonValley",
                "tools/verify_data_catalog.py",
            ],
        )
    )
    checks.extend(
        require_fragments(
            PARAMETER_ASSET_REPORT,
            [
                "Governed assets: 5",
                "Current locked-U0 assets: 4",
                "External-confirmation assets: 1",
                "Historical frozen assets: 0",
                "precomputed/stall_params_vatex_dino_v3.npz",
                "release/u0/params/comgenvid_region1_mean.npz",
                "Local sweep boundary",
            ],
        )
    )
    checks.extend(
        require_fragments(
            LOCAL_BRANCH_BOUNDARY_REPORT,
            [
                "score_local_raw",
                "0.1 * calibrated Patch Spatial + 0.9 * calibrated Patch D2",
                "src/patch_matching.py",
                "tests/test_local_branch.py",
                "0.8740724396350226/0.8722992273320395",
            ],
        )
    )
    checks.extend(
        require_fragments(
            GLOBAL_BRANCH_BOUNDARY_REPORT,
            [
                "score_global_raw",
                "0.5 * calibrated Global Spatial + 0.5 * calibrated Global T1",
                "src/stall.py",
                "tests/test_global_branch.py",
                "0.8740724396350226/0.8722992273320395",
            ],
        )
    )
    checks.extend(
        require_fragments(
            BACKBONE_BOUNDARY_REPORT,
            [
                "src/alpha_stalled/backbone.py",
                "checkpoint mtime_ns",
                "checkpoint bytes and content SHA-256",
                "tests/test_backbone.py",
                "0.8740724396350226/0.8722992273320395",
            ],
        )
    )
    checks.extend(
        require_fragments(
            VIDEO_IO_BOUNDARY_REPORT,
            [
                "src/alpha_stalled/video_io.py",
                "decode_indexed_frames",
                "require_all=True",
                "stall.load_video_frames",
                "tests/test_video_io.py",
                "0.8740724396350226/0.8722992273320395",
            ],
        )
    )
    checks.extend(
        require_fragments(
            U0_PROTOCOL_BOUNDARY_REPORT,
            [
                "alpha_stalled/u0_protocol.py",
                "alpha_stalled/u0_scoring.py",
                "alpha_stalled/u0_analysis.py",
                "alpha_stalled/metrics.py",
                "None of those three formal CLIs imports another file from `tools/`",
                "tests/test_u0_protocol.py",
                "tests/test_u0_pipeline.py",
                "tools/verify_u0_pipeline_reconstruction.py",
                "58,496 windows",
                "21,421 videos",
                "real video is the positive class",
                "0.8740724396350226/0.8722992273320395",
            ],
        )
    )
    checks.extend(
        require_fragments(
            SCORE_CSV_BOUNDARY_REPORT,
            [
                "src/alpha_stalled/score_csv.py",
                "tools/eval_alpha_stalled.py",
                "tools/eval_score_csv.py",
                "tools/fuse_scores.py",
                "same function objects",
                "alpha_stalled.u0_protocol",
            ],
        )
    )
    checks.extend(
        require_fragments(
            TOOL_DEPENDENCY_REPORT,
            [
                "Tool modules: 126",
                "Registered `tools -> tools` edges: 8",
                "Edges importing private symbols: 2",
                "Retained tool families: 4",
                "Dependency cycles: 0",
                "`formal_release` | 10",
                "`paper_evidence` | 45",
                "`governance` | 16",
                "`historical_frozen` | 38",
                "`research_utility` | 1",
                "`compatibility` | 16",
                "`compatibility_wrapper`: 14",
                "`retain_referenced`: 3",
                "Archived wrapper integrity",
                "Original implementation SHA-256",
                "Wrapper SHA-256",
                "Archived implementation SHA-256",
                "`retain_historical`: 8",
                "`frozen_historical`: 2",
                "`frozen_supplement`: 2",
                "`d3_operator_controls`",
                "`keyframe_case_visualization`",
                "`multiscale_intermediate_layers`",
                "`u0_robustness_injection`",
                "`evaluate_d3_robustness`",
                "`score_u0_injections`",
            ],
        )
    )
    checks.extend(
        require_fragments(
            CACHE_INVENTORY_REPORT,
            [
                "cache_inventory_20260814",
                "249764",
                "386.08 GiB",
                "P3_safe_delete_candidate",
                "manual approval",
            ],
        )
    )

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    require(
        "当前已验证的 release baseline" not in root_readme,
        "root README still presents the historical baseline as current",
    )
    checks.append("root README does not label the historical baseline as current")

    configs_readme = (ROOT / "configs/README.md").read_text(encoding="utf-8")
    config_assets = discover_config_assets(ROOT / "configs")
    unindexed_config_assets = sorted(
        path.removeprefix("configs/")
        for path in config_assets
        if path.removeprefix("configs/") not in configs_readme
    )
    require(
        not unindexed_config_assets,
        f"configs/README.md does not index config assets: {unindexed_config_assets}",
    )
    checks.append(f"config index covers {len(config_assets)} registered assets")

    release_index_summary = validate_release_indexes(ROOT / "release")
    checks.append(
        f"release indexes cover {release_index_summary.directory_count} directories and "
        f"{release_index_summary.asset_count} assets"
    )

    reports_readme = (ROOT / "reports/README.md").read_text(encoding="utf-8")
    top_level_reports = sorted(
        path for path in (ROOT / "reports").glob("*.md") if path.name != "README.md"
    )
    unindexed_reports = [
        path.name for path in top_level_reports if path.name not in reports_readme
    ]
    require(
        not unindexed_reports,
        f"reports/README.md does not index top-level reports: {unindexed_reports}",
    )
    checks.append(f"reports index covers {len(top_level_reports)} top-level reports")

    results_readme = (ROOT / "results/README.md").read_text(encoding="utf-8")
    results_root = ROOT / "results"
    top_level_result_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    unindexed_result_dirs = [
        path.name for path in top_level_result_dirs if path.name not in results_readme
    ]
    require(
        not unindexed_result_dirs,
        f"results/README.md does not index top-level directories: {unindexed_result_dirs}",
    )
    checks.append(
        f"results index covers {len(top_level_result_dirs)} top-level directories"
    )

    top_level_result_files = sorted(
        path for path in results_root.iterdir() if path.is_file() and path.name != "README.md"
    )
    unindexed_result_files = [
        path.name for path in top_level_result_files if path.name not in results_readme
    ]
    require(
        not unindexed_result_files,
        f"results/README.md does not index top-level files: {unindexed_result_files}",
    )
    checks.append(f"results index covers {len(top_level_result_files)} top-level files")

    stray_result_logs = sorted(
        str(path.relative_to(ROOT)) for path in results_root.rglob("*.log")
    )
    require(
        not stray_result_logs,
        f"runtime logs must live under logs/<experiment_id>, found: {stray_result_logs}",
    )
    checks.append("results tree contains no runtime logs")

    script_indexes = (
        (ROOT / "scripts", ROOT / "scripts/README.md"),
        (ROOT / "scripts/experiments", ROOT / "scripts/experiments/README.md"),
        (ROOT / "scripts/reproduce", ROOT / "scripts/reproduce/README.md"),
    )
    indexed_script_count = 0
    for script_dir, index_path in script_indexes:
        index_text = index_path.read_text(encoding="utf-8")
        scripts = sorted(script_dir.glob("*.sh"))
        missing = [path.name for path in scripts if path.name not in index_text]
        require(
            not missing,
            f"{index_path.relative_to(ROOT)} does not index scripts: {missing}",
        )
        indexed_script_count += len(scripts)
    checks.append(f"script indexes cover {indexed_script_count} shell entrypoints")

    shell_scripts = sorted((ROOT / "scripts").rglob("*.sh"))
    for script in shell_scripts:
        script_text = script.read_text(encoding="utf-8")
        require(
            script_text.startswith("#!/usr/bin/env bash\n"),
            f"shell script lacks the canonical Bash shebang: {script.relative_to(ROOT)}",
        )
        require(
            "set -euo pipefail" in script_text,
            f"shell script lacks strict mode: {script.relative_to(ROOT)}",
        )
    checks.append(f"{len(shell_scripts)} shell entrypoints use Bash strict mode")

    for path in DOCUMENTS:
        require(path.is_file(), f"missing project entry document: {path.relative_to(ROOT)}")
        checks.extend(verify_markdown_links(path))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = verify()
    payload = {"passed": True, "check_count": len(checks), "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"project documentation verification passed: {len(checks)} checks")


if __name__ == "__main__":
    main()
