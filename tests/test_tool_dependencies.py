from __future__ import annotations

import hashlib
import tempfile
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.tool_dependencies import (
    dependency_inventory,
    discover_tool_dependencies,
    read_tool_dependency_policy,
    render_dependency_report,
    validate_tool_dependencies,
)


class ToolDependencyTests(unittest.TestCase):
    @staticmethod
    def _complete_lifecycle_classes(
        historical: tuple[str, ...] = ("a", "b"),
    ) -> list[dict[str, object]]:
        assignments = {
            "formal_release": ("c",),
            "paper_evidence": ("d",),
            "governance": ("e",),
            "historical_frozen": historical,
            "research_utility": ("f",),
            "compatibility": ("g",),
        }
        return [
            {
                "lifecycle": lifecycle,
                "description": f"{lifecycle} description",
                "new_work_policy": f"{lifecycle} new work",
                "archive_policy": f"{lifecycle} archive",
                "tools": list(tools),
            }
            for lifecycle, tools in assignments.items()
        ]

    @staticmethod
    def _archive_maintenance_decisions() -> list[dict[str, object]]:
        return [
            {
                "tool": name,
                "decision": "archive_candidate",
                "reason": f"archive {name}",
                "archive_target": f"research_archive/tools/{name}.py",
                "evidence_paths": ["evidence.csv"],
            }
            for name in ("f", "g")
        ]

    def test_discovers_and_groups_imported_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("from b import z, x\nfrom b import y\n")
            (root / "b.py").write_text("VALUE = 1\n")
            edges = discover_tool_dependencies(root)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].key, ("a", "b"))
        self.assertEqual(edges[0].imported_symbols, ("x", "y", "z"))

    def test_rejects_unregistered_edge_and_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("from b import VALUE\n")
            (root / "b.py").write_text("from a import VALUE\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "allowed_edges": [],
            }
            _, errors = validate_tool_dependencies(root, policy)
        self.assertTrue(any("unregistered edge: a -> b" in error for error in errors))
        self.assertTrue(any("tool dependency cycle" in error for error in errors))

    def test_rejects_isolated_entrypoint_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("from b import VALUE\n")
            (root / "b.py").write_text("VALUE = 1\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": ["a"],
                "allowed_edges": [
                    {
                        "source": "a",
                        "target": "b",
                        "imported_symbols": ["VALUE"],
                        "category": "historical_experiment_family",
                        "migration_status": "migrate_to_shared",
                        "reason": "test",
                    }
                ],
            }
            _, errors = validate_tool_dependencies(root, policy)
        self.assertIn("isolated entrypoint a imports tools: b", errors)

    def test_retained_edge_requires_family_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in "abcdefg":
                (tools / f"{name}.py").write_text("VALUE = 1\n")
            (tools / "a.py").write_text("from b import VALUE\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "lifecycle_classes": self._complete_lifecycle_classes(),
                "retained_families": [],
                "allowed_edges": [
                    {
                        "source": "a",
                        "target": "b",
                        "imported_symbols": ["VALUE"],
                        "category": "historical_experiment_family",
                        "migration_status": "retain_historical",
                        "reason": "test",
                    }
                ],
            }
            _, errors = validate_tool_dependencies(tools, policy)
        self.assertTrue(any("has invalid family_id" in error for error in errors))

    def test_rejects_cross_lifecycle_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in "abcdefg":
                (tools / f"{name}.py").write_text("VALUE = 1\n")
            (tools / "c.py").write_text("from b import VALUE\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "lifecycle_classes": self._complete_lifecycle_classes(),
                "retained_families": [],
                "allowed_edges": [
                    {
                        "source": "c",
                        "target": "b",
                        "imported_symbols": ["VALUE"],
                        "category": "historical_experiment_family",
                        "migration_status": "migrate_to_shared",
                        "reason": "test",
                    }
                ],
            }
            _, errors = validate_tool_dependencies(tools, policy)
        self.assertTrue(any("cross-lifecycle tool dependency" in error for error in errors))
        self.assertIn("protected lifecycle entrypoint c imports tool b", errors)

    def test_inventory_hashes_retained_family_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in "abcdefg":
                (tools / f"{name}.py").write_text("VALUE = 1\n")
            (tools / "a.py").write_text("from b import VALUE\n")
            evidence = root / "evidence.csv"
            evidence.write_text("metric,value\nap,0.9\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "lifecycle_classes": self._complete_lifecycle_classes(),
                "maintenance_decisions": self._archive_maintenance_decisions(),
                "retained_families": [
                    {
                        "family_id": "demo",
                        "status": "frozen_historical",
                        "tools": ["a", "b"],
                        "evidence_paths": ["evidence.csv"],
                        "archive_target": "research_archive/tools/demo",
                        "reason": "test family",
                        "archive_gate": "test gate",
                    }
                ],
                "allowed_edges": [
                    {
                        "source": "a",
                        "target": "b",
                        "imported_symbols": ["VALUE"],
                        "category": "historical_experiment_family",
                        "migration_status": "retain_historical",
                        "family_id": "demo",
                        "reason": "test",
                    }
                ],
            }
            payload = dependency_inventory(tools, policy)
        self.assertEqual(payload["retained_family_count"], 1)
        item = payload["retained_families"][0]["evidence"][0]
        self.assertEqual(item["bytes"], len("metric,value\nap,0.9\n"))
        self.assertEqual(
            item["sha256"], hashlib.sha256(b"metric,value\nap,0.9\n").hexdigest()
        )

    def test_archive_candidate_rejects_new_repository_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in "abcdefg":
                (tools / f"{name}.py").write_text("VALUE = 1\n")
            (root / "evidence.csv").write_text("metric,value\nap,0.9\n")
            docs = root / "docs"
            docs.mkdir()
            (docs / "reference.md").write_text("Run tools/f.py for this workflow.\n")
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "lifecycle_classes": self._complete_lifecycle_classes(),
                "maintenance_decisions": self._archive_maintenance_decisions(),
                "retained_families": [],
                "allowed_edges": [],
            }
            _, errors = validate_tool_dependencies(tools, policy)
        self.assertTrue(any("archive candidate f still has references" in error for error in errors))

    def test_compatibility_wrapper_rejects_wrong_archive_forwarder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            for name in "abcdefg":
                (tools / f"{name}.py").write_text("VALUE = 1\n")
            (root / "evidence.csv").write_text("metric,value\nap,0.9\n")
            archive = root / "research_archive/tools"
            archive.mkdir(parents=True)
            (archive / "g.py").write_text("VALUE = 2\n")
            decisions = self._archive_maintenance_decisions()
            decisions[1] = {
                "tool": "g",
                "decision": "compatibility_wrapper",
                "reason": "archived implementation with old command compatibility",
                "archive_target": "research_archive/tools/g.py",
                "original_sha256": "0" * 64,
                "evidence_paths": ["evidence.csv"],
            }
            policy = {
                "schema_version": "alpha_stalled_tool_dependencies_v4",
                "tool_root": "tools",
                "isolated_entrypoints": [],
                "lifecycle_classes": self._complete_lifecycle_classes(),
                "maintenance_decisions": decisions,
                "retained_families": [],
                "allowed_edges": [],
            }
            _, errors = validate_tool_dependencies(tools, policy)
        self.assertTrue(
            any("compatibility_wrapper g does not forward" in error for error in errors)
        )

    def test_repository_policy_and_artifacts_are_exact(self) -> None:
        policy = read_tool_dependency_policy(ROOT / "configs/tool_dependencies.yaml")
        payload = dependency_inventory(ROOT / "tools", policy)
        self.assertEqual(payload["edge_count"], 8)
        self.assertEqual(payload["private_symbol_edge_count"], 2)
        self.assertEqual(payload["retained_family_count"], 4)
        self.assertEqual(
            payload["lifecycle_counts"],
            {
                "compatibility": 16,
                "formal_release": 10,
                "governance": 16,
                "historical_frozen": 38,
                "paper_evidence": 45,
                "research_utility": 1,
            },
        )
        self.assertEqual(len(payload["modules"]), 126)
        self.assertEqual(
            payload["maintenance_decision_counts"],
            {
                "compatibility_wrapper": 14,
                "retain_referenced": 3,
            },
        )
        wrappers = [
            row
            for row in payload["maintenance_decisions"]
            if row["decision"] == "compatibility_wrapper"
        ]
        self.assertEqual(len(wrappers), 14)
        self.assertTrue(all(row["compatibility_wrapper"] for row in wrappers))
        self.assertTrue(all(row["archived_implementation"] for row in wrappers))
        self.assertEqual(
            payload["retained_family_status_counts"],
            {"frozen_historical": 2, "frozen_supplement": 2},
        )
        self.assertTrue(all(edge["family_id"] for edge in payload["edges"]))
        stored = yaml.safe_load(
            (ROOT / "results/research_summary/tool_dependency_inventory.json").read_text()
        )
        self.assertEqual(stored, payload)
        self.assertEqual(
            (ROOT / "reports/tool_dependency_inventory.md").read_text(),
            render_dependency_report(payload),
        )


if __name__ == "__main__":
    unittest.main()
