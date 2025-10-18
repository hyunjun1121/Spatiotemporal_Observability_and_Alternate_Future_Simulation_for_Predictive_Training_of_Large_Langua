#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MOJIBAKE_PATTERNS = ["��", "�"]
SKIP_DIRS = {".git", "__pycache__", "runs", "plots"}


def is_text(path: Path) -> bool:
    return path.suffix not in {".png", ".pt", ".pdf", ".zip", ".tar", ".gz"}


def scan(root: Path) -> list[str]:
    warnings: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in SKIP_DIRS:
                continue
            if any(parent.name in SKIP_DIRS for parent in path.parents):
                continue
            continue
        if path.name == "check_encoding.py":
            continue
        if not is_text(path):
            continue
        if any(parent.name in SKIP_DIRS for parent in path.parents):
            continue
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.append(f"[encoding] {path} is not UTF-8")
            continue
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in data:
                warnings.append(f"[mojibake] {path} contains '{pattern}'")
                break
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check repository for UTF-8 encoding issues")
    parser.add_argument("root", nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = Path(args.root)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    targets = [root]
    paper_dir = Path("paper")
    try:
        root_resolved = root.resolve()
    except FileNotFoundError:
        root_resolved = root
    if paper_dir.exists():
        paper_resolved = paper_dir.resolve()
        include_paper = True
        try:
            paper_resolved.relative_to(root_resolved)
            include_paper = False
        except ValueError:
            include_paper = True
        if include_paper:
            targets.append(paper_resolved)
    warnings: list[str] = []
    for target in targets:
        warnings.extend(scan(target))
    if warnings:
        for warning in warnings:
            print(warning)
        sys.exit(1)
    print("All files passed UTF-8/mojibake check.")


if __name__ == "__main__":
    main()
