#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate standalone import boundaries for Agent_stock project."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_TEXT = (
    "/Users/hoaon/Desktop/毕设相关/project/v3/daily_stock_analysis",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "Agent_stock.",
)


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"} for part in path.parts):
            continue
        yield path


def main() -> int:
    violations: list[str] = []

    for path in iter_python_files(PROJECT_ROOT):
        if path == Path(__file__).resolve():
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
                    name = alias.name
                    if name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{rel}:{node.lineno} forbidden import: {name}")
            elif isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                mod = node.module
                if mod.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{rel}:{node.lineno} forbidden from-import: {mod}")

    if violations:
        print("Import boundary check FAILED:")
        for item in violations:
            print(f"- {item}")
        return 1

    print("Import boundary check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
