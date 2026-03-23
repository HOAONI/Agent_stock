#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理仓库内的本地运行产物与缓存目录。"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {".git", ".venv", "venv"}
ARTIFACT_DIR_NAMES = {"__pycache__", ".pytest_cache", "logs"}
ARTIFACT_FILE_PATHS = (
    PROJECT_ROOT / "data" / "stock_analysis.db",
)


def should_skip(path: Path) -> bool:
    """判断路径是否位于应跳过的目录中。"""
    return any(part in SKIP_PARTS for part in path.parts)


def iter_artifact_dirs(root: Path):
    """遍历需要删除的工作区产物目录。"""
    for path in root.rglob("*"):
        if should_skip(path):
            continue
        if path.is_dir() and path.name in ARTIFACT_DIR_NAMES:
            yield path


def remove_dir(path: Path) -> None:
    """删除目录并输出结果。"""
    if path.exists():
        shutil.rmtree(path)
        print(f"removed dir: {path.relative_to(PROJECT_ROOT)}")


def remove_file(path: Path) -> None:
    """删除文件并输出结果。"""
    if path.exists():
        path.unlink()
        print(f"removed file: {path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    """清理本地工作区产物。"""
    dirs_to_remove = sorted(
        {path for path in iter_artifact_dirs(PROJECT_ROOT)},
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    )

    removed = False
    for artifact_dir in dirs_to_remove:
        if artifact_dir.exists():
            remove_dir(artifact_dir)
            removed = True

    for artifact_file in ARTIFACT_FILE_PATHS:
        if artifact_file.exists():
            remove_file(artifact_file)
            removed = True

    if not removed:
        print("no workspace artifacts found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
