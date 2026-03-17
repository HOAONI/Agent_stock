#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查导入边界与工作区产物是否符合约束。"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELF_PATH = Path(__file__).resolve()
SKIP_PARTS = {".git", ".venv", "venv"}
FORBIDDEN_TEXT = (
    "/Users/hoaon/Desktop/毕设相关/project/v3/daily_stock_analysis",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "src",
    "bot",
    "Agent_stock",
)
DELETED_PATHS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "bot",
    PROJECT_ROOT / "agent_stock" / "integrations",
    PROJECT_ROOT / "chat-history.md",
    PROJECT_ROOT / "docs" / "CHANGELOG.md",
    PROJECT_ROOT / "tests" / "test_src_compat_imports.py",
)
ARTIFACT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "logs",
}
ARTIFACT_FILE_PATHS = (
    PROJECT_ROOT / "data" / "stock_analysis.db",
)
TRACKED_ARTIFACT_PREFIXES = (
    "__pycache__/",
    ".pytest_cache/",
    "logs/",
    "data/stock_analysis.db",
)


def iter_python_files(root: Path):
    """遍历项目中的 Python 文件，跳过虚拟环境和缓存目录。"""
    for path in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        yield path


def import_is_forbidden(name: str) -> bool:
    """判断导入路径是否仍指向已废弃模块前缀。"""
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_IMPORT_PREFIXES)


def iter_artifact_dirs(root: Path):
    """枚举需要清理的工作区产物目录。"""
    for path in root.rglob("*"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.is_dir() and path.name in ARTIFACT_DIR_NAMES:
            yield path


def iter_tracked_artifacts(root: Path):
    """查询 Git 中是否错误跟踪了缓存或运行产物。"""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    tracked = []
    for line in result.stdout.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if normalized in TRACKED_ARTIFACT_PREFIXES:
            tracked.append(normalized)
            continue
        if any(normalized.startswith(prefix) for prefix in TRACKED_ARTIFACT_PREFIXES if prefix.endswith("/")):
            tracked.append(normalized)
    return tracked


def main() -> int:
    """执行所有导入边界与工作区卫生检查。"""
    violations: list[str] = []

    for deleted_path in DELETED_PATHS:
        if deleted_path.exists():
            violations.append(f"{deleted_path.relative_to(PROJECT_ROOT)}: legacy path should not exist")

    for artifact_dir in iter_artifact_dirs(PROJECT_ROOT):
        violations.append(f"{artifact_dir.relative_to(PROJECT_ROOT)}: workspace artifact should be removed")

    for artifact_file in ARTIFACT_FILE_PATHS:
        if artifact_file.exists():
            violations.append(f"{artifact_file.relative_to(PROJECT_ROOT)}: workspace artifact should be removed")

    for tracked_path in iter_tracked_artifacts(PROJECT_ROOT):
        violations.append(f"{tracked_path}: artifact path must not be tracked")

    for path in iter_python_files(PROJECT_ROOT):
        if path == SELF_PATH:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(PROJECT_ROOT)

        for marker in FORBIDDEN_TEXT:
            if marker in text:
                violations.append(f"{rel}: contains forbidden external path reference")

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error: {exc}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if import_is_forbidden(alias.name):
                        violations.append(f"{rel}:{node.lineno} forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and import_is_forbidden(node.module):
                    violations.append(f"{rel}:{node.lineno} forbidden from-import: {node.module}")

    if violations:
        print("Import boundary check FAILED:")
        for item in sorted(set(violations)):
            print(f"- {item}")
        return 1

    print("Import boundary check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
