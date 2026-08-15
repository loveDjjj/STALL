"""AST-based dependency governance for repository command-line tools."""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from .release_io import sha256_file


SCHEMA_VERSION = "alpha_stalled_tool_dependencies_v4"
INVENTORY_SCHEMA_VERSION = "alpha_stalled_tool_dependency_inventory_v4"
ALLOWED_CATEGORIES = {
    "duration_aware_extension",
    "historical_d3",
    "historical_experiment_family",
    "historical_scoring",
    "research_visualization",
    "u0_extension",
}
ALLOWED_MIGRATION_STATUSES = {"retain_historical", "migrate_to_shared"}
ALLOWED_FAMILY_STATUSES = {"frozen_historical", "frozen_supplement"}
ALLOWED_LIFECYCLES = {
    "compatibility",
    "formal_release",
    "governance",
    "historical_frozen",
    "paper_evidence",
    "research_utility",
}
ALLOWED_MAINTENANCE_DECISIONS = {
    "archive_candidate",
    "compatibility_wrapper",
    "replace_before_archive",
    "retain_referenced",
}
MAINTENANCE_LIFECYCLES = {"compatibility", "research_utility"}
REFERENCE_ROOTS = (
    "configs",
    "docs",
    "paper",
    "release",
    "reports",
    "research_archive",
    "results/research_summary",
    "scripts",
    "src",
    "tests",
    "tools",
)
REFERENCE_SUFFIXES = {
    ".csv",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FAMILY_LIFECYCLE = {
    "frozen_historical": "historical_frozen",
    "frozen_supplement": "paper_evidence",
}


@dataclass(frozen=True, order=True)
class ToolDependency:
    source: str
    target: str
    imported_symbols: tuple[str, ...]
    line: int

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.target


def tool_modules(tool_root: Path) -> dict[str, Path]:
    """Return every top-level Python tool keyed by its importable stem."""

    modules = {path.stem: path for path in sorted(tool_root.glob("*.py"))}
    if len(modules) != len(list(tool_root.glob("*.py"))):
        raise ValueError("tool module stems are not unique")
    return modules


def discover_tool_dependencies(tool_root: Path) -> list[ToolDependency]:
    """Parse imports and return direct dependencies between top-level tools."""

    modules = tool_modules(tool_root)
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for source, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imports: list[tuple[str, tuple[str, ...]]] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.append(
                    (
                        node.module.split(".", 1)[0],
                        tuple(alias.name for alias in node.names),
                    )
                )
            elif isinstance(node, ast.Import):
                imports.extend(
                    (alias.name.split(".", 1)[0], ("*",)) for alias in node.names
                )
            for target, symbols in imports:
                if target not in modules:
                    continue
                key = (source, target)
                record = grouped.setdefault(
                    key, {"symbols": set(), "line": int(node.lineno)}
                )
                record["symbols"].update(symbols)  # type: ignore[union-attr]
                record["line"] = min(int(record["line"]), int(node.lineno))
    return [
        ToolDependency(
            source=source,
            target=target,
            imported_symbols=tuple(sorted(record["symbols"])),
            line=int(record["line"]),
        )
        for (source, target), record in sorted(grouped.items())
    ]


def read_tool_dependency_policy(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tool dependency policy must be a mapping")
    return payload


def _configured_edges(policy: dict) -> dict[tuple[str, str], dict]:
    configured: dict[tuple[str, str], dict] = {}
    for row in policy.get("allowed_edges", []):
        key = str(row.get("source", "")), str(row.get("target", ""))
        if key in configured:
            raise ValueError(f"duplicate configured tool dependency: {key}")
        configured[key] = row
    return configured


def _configured_families(policy: dict) -> dict[str, dict]:
    configured: dict[str, dict] = {}
    for row in policy.get("retained_families", []):
        family_id = str(row.get("family_id", ""))
        if not family_id:
            raise ValueError("retained family is missing family_id")
        if family_id in configured:
            raise ValueError(f"duplicate retained family: {family_id}")
        configured[family_id] = row
    return configured


def _configured_lifecycles(policy: dict) -> dict[str, dict]:
    configured: dict[str, dict] = {}
    for row in policy.get("lifecycle_classes", []):
        lifecycle = str(row.get("lifecycle", ""))
        if not lifecycle:
            raise ValueError("lifecycle class is missing lifecycle")
        if lifecycle in configured:
            raise ValueError(f"duplicate lifecycle class: {lifecycle}")
        configured[lifecycle] = row
    return configured


def _configured_maintenance_decisions(policy: dict) -> dict[str, dict]:
    configured: dict[str, dict] = {}
    for row in policy.get("maintenance_decisions", []):
        tool = str(row.get("tool", ""))
        if not tool:
            raise ValueError("maintenance decision is missing tool")
        if tool in configured:
            raise ValueError(f"duplicate maintenance decision: {tool}")
        configured[tool] = row
    return configured


def discover_tool_references(
    repository_root: Path, tool_names: set[str]
) -> dict[str, list[dict[str, object]]]:
    """Find deterministic non-result textual references to selected tool names."""

    references: dict[str, list[dict[str, object]]] = {
        name: [] for name in sorted(tool_names)
    }
    patterns = {
        name: re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?:\.py)?(?![A-Za-z0-9_])"
        )
        for name in tool_names
    }
    candidates = [
        path
        for path in sorted(repository_root.glob("*"))
        if path.is_file() and path.suffix.lower() in REFERENCE_SUFFIXES
    ]
    for root_name in REFERENCE_ROOTS:
        root = repository_root / root_name
        if root.is_dir():
            candidates.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in REFERENCE_SUFFIXES
            )
    excluded = {
        repository_root / "configs/tool_dependencies.yaml",
        repository_root / "reports/tool_dependency_inventory.md",
        repository_root
        / "results/research_summary/tool_dependency_inventory.json",
    }
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or path in excluded:
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(repository_root).as_posix()
        lines = text.splitlines()
        for name in sorted(tool_names):
            pattern = patterns[name]
            if (
                path == repository_root / "tools" / f"{name}.py"
                or not pattern.search(text)
            ):
                continue
            line_numbers = [
                index
                for index, line in enumerate(lines, start=1)
                if pattern.search(line)
            ]
            references[name].append({"path": relative, "lines": line_numbers})
    return references


def _find_cycle(modules: set[str], edges: list[ToolDependency]) -> list[str] | None:
    adjacency = {module: [] for module in modules}
    for edge in edges:
        adjacency[edge.source].append(edge.target)
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(module: str) -> list[str] | None:
        if module in active_set:
            start = active.index(module)
            return [*active[start:], module]
        if module in visited:
            return None
        active.append(module)
        active_set.add(module)
        for target in sorted(adjacency[module]):
            cycle = visit(target)
            if cycle:
                return cycle
        active.pop()
        active_set.remove(module)
        visited.add(module)
        return None

    for module in sorted(modules):
        cycle = visit(module)
        if cycle:
            return cycle
    return None


def validate_tool_dependencies(
    tool_root: Path, policy: dict
) -> tuple[list[ToolDependency], list[str]]:
    """Validate current AST edges against an explicit repository policy."""

    errors: list[str] = []
    modules = tool_modules(tool_root)
    edges = discover_tool_dependencies(tool_root)
    actual = {edge.key: edge for edge in edges}
    try:
        configured = _configured_edges(policy)
        families = _configured_families(policy)
        lifecycles = _configured_lifecycles(policy)
        maintenance = _configured_maintenance_decisions(policy)
    except ValueError as error:
        return edges, [str(error)]
    if policy.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"wrong schema_version: {policy.get('schema_version')!r}")
    if policy.get("tool_root") != "tools":
        errors.append("tool_root must be 'tools'")

    missing_lifecycles = sorted(ALLOWED_LIFECYCLES - set(lifecycles))
    unexpected_lifecycles = sorted(set(lifecycles) - ALLOWED_LIFECYCLES)
    if missing_lifecycles:
        errors.append(f"missing lifecycle classes: {missing_lifecycles}")
    if unexpected_lifecycles:
        errors.append(f"invalid lifecycle classes: {unexpected_lifecycles}")
    tool_lifecycle: dict[str, str] = {}
    for lifecycle, row in sorted(lifecycles.items()):
        for field in ("description", "new_work_policy", "archive_policy"):
            if not str(row.get(field, "")).strip():
                errors.append(f"lifecycle {lifecycle} is missing {field}")
        lifecycle_tools = [str(value) for value in row.get("tools", [])]
        if not lifecycle_tools:
            errors.append(f"lifecycle {lifecycle} has no tools")
        if len(lifecycle_tools) != len(set(lifecycle_tools)):
            errors.append(f"duplicate tool in lifecycle {lifecycle}")
        for name in lifecycle_tools:
            if name not in modules:
                errors.append(f"lifecycle {lifecycle} references missing tool: {name}")
            previous = tool_lifecycle.get(name)
            if previous is not None:
                errors.append(
                    f"tool {name} belongs to multiple lifecycles: {previous}, {lifecycle}"
                )
            tool_lifecycle[name] = lifecycle
    unclassified = sorted(set(modules) - set(tool_lifecycle))
    unknown_tools = sorted(set(tool_lifecycle) - set(modules))
    if unclassified:
        errors.append(f"unclassified tool modules: {unclassified}")
    if unknown_tools:
        errors.append(f"classified tools do not exist: {unknown_tools}")

    maintenance_tools = {
        name
        for name, lifecycle in tool_lifecycle.items()
        if lifecycle in MAINTENANCE_LIFECYCLES
    }
    missing_decisions = sorted(maintenance_tools - set(maintenance))
    unexpected_decisions = sorted(set(maintenance) - maintenance_tools)
    if missing_decisions:
        errors.append(f"tools missing maintenance decisions: {missing_decisions}")
    if unexpected_decisions:
        errors.append(
            f"maintenance decisions cover ineligible tools: {unexpected_decisions}"
        )
    reference_map = discover_tool_references(tool_root.parent, maintenance_tools)
    for name, row in sorted(maintenance.items()):
        decision = str(row.get("decision", ""))
        if decision not in ALLOWED_MAINTENANCE_DECISIONS:
            errors.append(f"invalid maintenance decision for {name}: {decision!r}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"missing maintenance reason for {name}")
        references = reference_map.get(name, [])
        if decision == "archive_candidate" and references:
            errors.append(
                f"archive candidate {name} still has references: "
                f"{[item['path'] for item in references]}"
            )
        if decision in {"replace_before_archive", "retain_referenced"} and not references:
            errors.append(f"{decision} tool {name} has no repository references")
        if decision == "replace_before_archive" and not str(
            row.get("replacement", "")
        ).strip():
            errors.append(f"replace_before_archive tool {name} has no replacement")
        if decision in {
            "archive_candidate",
            "compatibility_wrapper",
            "replace_before_archive",
        }:
            archive_target = Path(str(row.get("archive_target", "")))
            if (
                not str(row.get("archive_target", "")).strip()
                or archive_target.is_absolute()
                or ".." in archive_target.parts
            ):
                errors.append(f"invalid maintenance archive_target for {name}")
        evidence_paths = [str(value) for value in row.get("evidence_paths", [])]
        if decision in {"archive_candidate", "compatibility_wrapper"} and not evidence_paths:
            errors.append(f"{decision} tool {name} has no evidence paths")
        if len(evidence_paths) != len(set(evidence_paths)):
            errors.append(f"duplicate maintenance evidence path for {name}")
        for value in evidence_paths:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(
                    f"maintenance evidence path for {name} must be repository-relative: "
                    f"{value}"
                )
            elif not (tool_root.parent / relative).is_file():
                errors.append(
                    f"maintenance evidence file for {name} does not exist: {value}"
                )
        if decision == "compatibility_wrapper":
            if tool_lifecycle.get(name) != "compatibility":
                errors.append(
                    f"compatibility_wrapper tool {name} must use compatibility lifecycle"
                )
            target_relative = Path(str(row.get("archive_target", "")))
            target = tool_root.parent / target_relative
            if not target.is_file():
                errors.append(
                    f"compatibility_wrapper archive implementation is missing for {name}: "
                    f"{target_relative}"
                )
            archive_root = Path("research_archive/tools")
            try:
                wrapper_target = target_relative.relative_to(archive_root).as_posix()
            except ValueError:
                errors.append(
                    f"compatibility_wrapper target for {name} is outside "
                    f"{archive_root}: {target_relative}"
                )
            else:
                wrapper_text = (tool_root / f"{name}.py").read_text(encoding="utf-8")
                if "run_archived_tool" not in wrapper_text or wrapper_target not in wrapper_text:
                    errors.append(
                        f"compatibility_wrapper {name} does not forward to {wrapper_target}"
                    )
            original_sha256 = str(row.get("original_sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", original_sha256):
                errors.append(
                    f"compatibility_wrapper {name} has invalid original_sha256"
                )

    family_membership: dict[str, str] = {}
    for family_id, row in sorted(families.items()):
        if row.get("status") not in ALLOWED_FAMILY_STATUSES:
            errors.append(f"invalid status for retained family {family_id}")
        family_tools = [str(value) for value in row.get("tools", [])]
        if len(family_tools) < 2:
            errors.append(f"retained family {family_id} must contain at least two tools")
        if len(family_tools) != len(set(family_tools)):
            errors.append(f"duplicate tool in retained family {family_id}")
        for name in family_tools:
            if name not in modules:
                errors.append(f"retained family {family_id} references missing tool: {name}")
            previous = family_membership.get(name)
            if previous is not None:
                errors.append(
                    f"tool {name} belongs to multiple retained families: "
                    f"{previous}, {family_id}"
                )
            family_membership[name] = family_id
        evidence_paths = [str(value) for value in row.get("evidence_paths", [])]
        if not evidence_paths:
            errors.append(f"retained family {family_id} has no evidence paths")
        if len(evidence_paths) != len(set(evidence_paths)):
            errors.append(f"duplicate evidence path in retained family {family_id}")
        for value in evidence_paths:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(
                    f"retained family {family_id} evidence path must be repository-relative: "
                    f"{value}"
                )
            elif not (tool_root.parent / relative).is_file():
                errors.append(
                    f"retained family {family_id} evidence file does not exist: {value}"
                )
        archive_target = Path(str(row.get("archive_target", "")))
        if (
            not str(row.get("archive_target", "")).strip()
            or archive_target.is_absolute()
            or ".." in archive_target.parts
        ):
            errors.append(f"invalid archive_target for retained family {family_id}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"missing reason for retained family {family_id}")
        if not str(row.get("archive_gate", "")).strip():
            errors.append(f"missing archive_gate for retained family {family_id}")
        expected_lifecycle = FAMILY_LIFECYCLE.get(str(row.get("status", "")))
        if expected_lifecycle is not None:
            for name in family_tools:
                actual_lifecycle = tool_lifecycle.get(name)
                if actual_lifecycle != expected_lifecycle:
                    errors.append(
                        f"retained family {family_id} requires lifecycle "
                        f"{expected_lifecycle} for {name}, got {actual_lifecycle}"
                    )

    for name in policy.get("isolated_entrypoints", []):
        if name not in modules:
            errors.append(f"isolated entrypoint does not exist: {name}")
        outgoing = [edge for edge in edges if edge.source == name]
        if outgoing:
            targets = ", ".join(edge.target for edge in outgoing)
            errors.append(f"isolated entrypoint {name} imports tools: {targets}")

    missing = sorted(set(configured) - set(actual))
    unexpected = sorted(set(actual) - set(configured))
    errors.extend(f"configured edge is absent: {source} -> {target}" for source, target in missing)
    errors.extend(f"unregistered edge: {source} -> {target}" for source, target in unexpected)

    for key in sorted(set(configured) & set(actual)):
        row = configured[key]
        source, target = key
        symbols = tuple(sorted(str(value) for value in row.get("imported_symbols", [])))
        if symbols != actual[key].imported_symbols:
            errors.append(
                f"imported symbols changed for {source} -> {target}: "
                f"configured={symbols} actual={actual[key].imported_symbols}"
            )
        if source not in modules or target not in modules:
            errors.append(f"configured edge references missing module: {source} -> {target}")
        if row.get("category") not in ALLOWED_CATEGORIES:
            errors.append(f"invalid category for {source} -> {target}")
        if row.get("migration_status") not in ALLOWED_MIGRATION_STATUSES:
            errors.append(f"invalid migration_status for {source} -> {target}")
        if not str(row.get("reason", "")).strip():
            errors.append(f"missing reason for {source} -> {target}")
        family_id = str(row.get("family_id", ""))
        if row.get("migration_status") == "retain_historical":
            if family_id not in families:
                errors.append(
                    f"retained edge {source} -> {target} has invalid family_id: "
                    f"{family_id!r}"
                )
            else:
                family_tools = {str(value) for value in families[family_id].get("tools", [])}
                if source not in family_tools or target not in family_tools:
                    errors.append(
                        f"retained edge {source} -> {target} is not contained in family "
                        f"{family_id}"
                    )
        elif family_id:
            errors.append(
                f"migratable edge {source} -> {target} must not declare family_id"
            )
        source_lifecycle = tool_lifecycle.get(source)
        target_lifecycle = tool_lifecycle.get(target)
        if source_lifecycle != target_lifecycle:
            errors.append(
                f"cross-lifecycle tool dependency {source} ({source_lifecycle}) -> "
                f"{target} ({target_lifecycle})"
            )

    family_edge_tools: dict[str, set[str]] = {family_id: set() for family_id in families}
    for row in configured.values():
        family_id = str(row.get("family_id", ""))
        if family_id in family_edge_tools:
            family_edge_tools[family_id].update(
                (str(row.get("source", "")), str(row.get("target", "")))
            )
    for family_id, row in sorted(families.items()):
        configured_tools = {str(value) for value in row.get("tools", [])}
        if not family_edge_tools[family_id]:
            errors.append(f"retained family {family_id} is not used by any edge")
        elif configured_tools != family_edge_tools[family_id]:
            missing_tools = sorted(configured_tools - family_edge_tools[family_id])
            extra_tools = sorted(family_edge_tools[family_id] - configured_tools)
            errors.append(
                f"retained family {family_id} tool/edge coverage mismatch: "
                f"tools_without_edges={missing_tools} edge_tools_not_declared={extra_tools}"
            )

    protected_lifecycles = {"formal_release", "governance"}
    for edge in edges:
        if tool_lifecycle.get(edge.source) in protected_lifecycles:
            errors.append(
                f"protected lifecycle entrypoint {edge.source} imports tool {edge.target}"
            )

    cycle = _find_cycle(set(modules), edges)
    if cycle:
        errors.append(f"tool dependency cycle: {' -> '.join(cycle)}")
    return edges, errors


def dependency_inventory(tool_root: Path, policy: dict) -> dict:
    """Build a deterministic machine-readable snapshot of the validated graph."""

    edges, errors = validate_tool_dependencies(tool_root, policy)
    if errors:
        raise ValueError("; ".join(errors))
    configured = _configured_edges(policy)
    families = _configured_families(policy)
    lifecycles = _configured_lifecycles(policy)
    maintenance = _configured_maintenance_decisions(policy)
    tool_lifecycle = {
        str(name): lifecycle
        for lifecycle, row in lifecycles.items()
        for name in row["tools"]
    }
    categories = Counter(configured[edge.key]["category"] for edge in edges)
    statuses = Counter(configured[edge.key]["migration_status"] for edge in edges)
    family_statuses = Counter(row["status"] for row in families.values())
    lifecycle_counts = Counter(tool_lifecycle.values())
    maintenance_counts = Counter(row["decision"] for row in maintenance.values())
    reference_map = discover_tool_references(tool_root.parent, set(maintenance))
    private_edges = [
        edge
        for edge in edges
        if any(symbol.startswith("_") and symbol != "*" for symbol in edge.imported_symbols)
    ]
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "policy_schema_version": policy["schema_version"],
        "tool_root": "tools",
        "tool_module_count": len(tool_modules(tool_root)),
        "edge_count": len(edges),
        "source_module_count": len({edge.source for edge in edges}),
        "target_module_count": len({edge.target for edge in edges}),
        "private_symbol_edge_count": len(private_edges),
        "retained_family_count": len(families),
        "isolated_entrypoints": sorted(policy.get("isolated_entrypoints", [])),
        "category_counts": dict(sorted(categories.items())),
        "migration_status_counts": dict(sorted(statuses.items())),
        "retained_family_status_counts": dict(sorted(family_statuses.items())),
        "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "maintenance_decision_counts": dict(sorted(maintenance_counts.items())),
        "maintenance_decisions": [
            {
                "tool": name,
                "lifecycle": tool_lifecycle[name],
                "decision": row["decision"],
                "reason": row["reason"],
                "replacement": row.get("replacement"),
                "archive_target": row.get("archive_target"),
                "original_sha256": row.get("original_sha256"),
                "compatibility_wrapper": (
                    {
                        "path": f"tools/{name}.py",
                        "bytes": (tool_root / f"{name}.py").stat().st_size,
                        "sha256": sha256_file(tool_root / f"{name}.py"),
                    }
                    if row["decision"] == "compatibility_wrapper"
                    else None
                ),
                "archived_implementation": (
                    {
                        "path": row["archive_target"],
                        "bytes": (tool_root.parent / row["archive_target"]).stat().st_size,
                        "sha256": sha256_file(tool_root.parent / row["archive_target"]),
                    }
                    if row["decision"] == "compatibility_wrapper"
                    else None
                ),
                "references": reference_map[name],
                "evidence": [
                    {
                        "path": value,
                        "bytes": (tool_root.parent / value).stat().st_size,
                        "sha256": sha256_file(tool_root.parent / value),
                    }
                    for value in sorted(
                        str(path) for path in row.get("evidence_paths", [])
                    )
                ],
            }
            for name, row in sorted(maintenance.items())
        ],
        "lifecycle_classes": [
            {
                "lifecycle": lifecycle,
                "description": row["description"],
                "new_work_policy": row["new_work_policy"],
                "archive_policy": row["archive_policy"],
                "tools": sorted(str(value) for value in row["tools"]),
            }
            for lifecycle, row in sorted(lifecycles.items())
        ],
        "modules": [
            {
                "name": name,
                "lifecycle": tool_lifecycle[name],
                "outgoing_tool_dependencies": sorted(
                    edge.target for edge in edges if edge.source == name
                ),
            }
            for name in sorted(tool_modules(tool_root))
        ],
        "retained_families": [
            {
                "family_id": family_id,
                "status": row["status"],
                "tools": sorted(str(value) for value in row["tools"]),
                "evidence": [
                    {
                        "path": value,
                        "bytes": (tool_root.parent / value).stat().st_size,
                        "sha256": sha256_file(tool_root.parent / value),
                    }
                    for value in sorted(str(path) for path in row["evidence_paths"])
                ],
                "archive_target": row["archive_target"],
                "reason": row["reason"],
                "archive_gate": row["archive_gate"],
            }
            for family_id, row in sorted(families.items())
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "line": edge.line,
                "imported_symbols": list(edge.imported_symbols),
                "category": configured[edge.key]["category"],
                "migration_status": configured[edge.key]["migration_status"],
                "family_id": configured[edge.key].get("family_id"),
                "reason": configured[edge.key]["reason"],
            }
            for edge in edges
        ],
    }


def render_dependency_report(inventory: dict) -> str:
    """Render the deterministic inventory as a concise Markdown governance report."""

    categories = "\n".join(
        f"- `{name}`: {count}" for name, count in inventory["category_counts"].items()
    )
    statuses = "\n".join(
        f"- `{name}`: {count}"
        for name, count in inventory["migration_status_counts"].items()
    )
    family_statuses = "\n".join(
        f"- `{name}`: {count}"
        for name, count in inventory["retained_family_status_counts"].items()
    )
    lifecycle_rows = "\n".join(
        "| `{lifecycle}` | {count} | {description} |".format(
            lifecycle=row["lifecycle"],
            count=inventory["lifecycle_counts"][row["lifecycle"]],
            description=row["description"],
        )
        for row in inventory["lifecycle_classes"]
    )
    lifecycle_sections = "\n\n".join(
        "\n".join(
            [
                f"### `{row['lifecycle']}`",
                "",
                f"- New work: {row['new_work_policy']}",
                f"- Archive: {row['archive_policy']}",
                "- Tools: " + ", ".join(f"`{name}`" for name in row["tools"]),
            ]
        )
        for row in inventory["lifecycle_classes"]
    )
    maintenance_counts = "\n".join(
        f"- `{name}`: {count}"
        for name, count in inventory["maintenance_decision_counts"].items()
    )
    maintenance_rows = "\n".join(
        "| `{tool}` | `{lifecycle}` | `{decision}` | {references} | `{target}` |".format(
            tool=row["tool"],
            lifecycle=row["lifecycle"],
            decision=row["decision"],
            references=len(row["references"]),
            target=row["archive_target"] or "-",
        )
        for row in inventory["maintenance_decisions"]
    )
    archived_wrapper_rows = "\n".join(
        "| `{tool}` | `{original}` | `{wrapper}` | `{archived}` |".format(
            tool=row["tool"],
            original=row["original_sha256"],
            wrapper=row["compatibility_wrapper"]["sha256"],
            archived=row["archived_implementation"]["sha256"],
        )
        for row in inventory["maintenance_decisions"]
        if row["decision"] == "compatibility_wrapper"
    )
    families = "\n\n".join(
        "\n".join(
            [
                f"### `{family['family_id']}`",
                "",
                f"- Status: `{family['status']}`",
                "- Tools: " + ", ".join(f"`{name}`" for name in family["tools"]),
                f"- Archive target: `{family['archive_target']}`",
                f"- Reason: {family['reason']}",
                f"- Archive gate: {family['archive_gate']}",
                "- Evidence:",
                *[
                    f"  - `{item['path']}` ({item['bytes']} bytes, sha256 `{item['sha256']}`)"
                    for item in family["evidence"]
                ],
            ]
        )
        for family in inventory["retained_families"]
    )
    rows = "\n".join(
        "| `{source}` | `{target}` | {symbols} | `{category}` | `{status}` | `{family}` |".format(
            source=edge["source"],
            target=edge["target"],
            symbols=", ".join(f"`{symbol}`" for symbol in edge["imported_symbols"]),
            category=edge["category"],
            status=edge["migration_status"],
            family=edge["family_id"] or "-",
        )
        for edge in inventory["edges"]
    )
    return f"""# Tool dependency inventory

This report is generated from `configs/tool_dependencies.yaml` and the current AST of
top-level Python files in `tools/`. It is a governance snapshot, not an execution graph.

## Summary

- Tool modules: {inventory['tool_module_count']}
- Registered `tools -> tools` edges: {inventory['edge_count']}
- Source modules with internal imports: {inventory['source_module_count']}
- Imported target modules: {inventory['target_module_count']}
- Edges importing private symbols: {inventory['private_symbol_edge_count']}
- Retained tool families: {inventory['retained_family_count']}
- Dependency cycles: 0

Protected formal, governance, and migrated historical entrypoints listed under
`isolated_entrypoints` have no internal tool imports. Any new edge, removed edge, symbol change,
or cycle fails verification.

## Tool lifecycle

| Lifecycle | Tools | Meaning |
|---|---:|---|
{lifecycle_rows}

Every top-level Python tool belongs to exactly one lifecycle. Formal release and governance tools
must not import another CLI, and all remaining internal edges must stay within one lifecycle.

{lifecycle_sections}

## Maintenance decisions

{maintenance_counts}

Reference counts scan non-result repository text while excluding this policy, its generated
artifacts, and each tool's own source file. Archive candidates must have zero references and
content-addressed evidence; compatibility wrappers preserve old command paths after physical
archival; referenced candidates must name a replacement before moving.

| Tool | Lifecycle | Decision | Reference files | Archive target |
|---|---|---|---:|---|
{maintenance_rows}

### Archived wrapper integrity

The original implementation hash is preserved from immediately before the move. Wrapper and
archived implementation hashes are recomputed from the current files on every inventory build.

| Tool | Original implementation SHA-256 | Wrapper SHA-256 | Archived implementation SHA-256 |
|---|---|---|---|
{archived_wrapper_rows}

## Categories

{categories}

## Migration status

{statuses}

`migrate_to_shared` marks reusable behavior that should move into `src/alpha_stalled/`.
`retain_historical` marks tightly coupled historical evidence that should remain stable until its
whole experiment family is archived.

## Retained family status

{family_statuses}

Every retained family owns all tools at the endpoints of its registered edges. Evidence files are
content-addressed in the JSON inventory. Physical moves are forbidden until the recorded archive
gate is met.

## Retained families

{families}

## Registered edges

| Source | Target | Imported symbols | Category | Migration | Family |
|---|---|---|---|---|---|
{rows}
"""


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ToolDependency",
    "dependency_inventory",
    "discover_tool_references",
    "discover_tool_dependencies",
    "read_tool_dependency_policy",
    "render_dependency_report",
    "tool_modules",
    "validate_tool_dependencies",
]
