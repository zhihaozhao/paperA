#!/usr/bin/env python3
"""
Collate all Markdown files into classified folders under docs/collated and
generate a manifest listing.

Categories (by simple filename/path heuristics):
  - daily/: daily notes and summaries
  - figures/: plotting and figure docs
  - experiments/: experiment summaries, execution notes, results
  - paper/: paper-related planning and section outlines
  - misc/: uncategorized

Writes:
  - docs/collated/<category>/*.md (copied)
  - docs/collated/MANIFEST.md
"""
from __future__ import annotations

import shutil
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]

SRC_ROOTS = [REPO]
OUT_ROOT = REPO / "docs" / "collated"

def find_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in SRC_ROOTS:
        for p in root.rglob("*.md"):
            # Skip collated outputs and virtualenvs or git folders
            rel = p.relative_to(REPO)
            parts = rel.parts
            if parts[:2] == ("docs", "collated"):
                continue
            if any(seg.startswith(".") for seg in parts):
                continue
            files.append(p)
    return files

def classify(path: Path) -> str:
    rel = path.relative_to(REPO).as_posix().lower()
    name = path.name.lower()
    if rel.startswith("docs/daily/") or re.search(r"aug\d{1,2}|daily|summary", name):
        return "daily"
    if rel.startswith("paper/") or "paper" in rel:
        return "paper"
    if "figure" in rel or "figures" in rel:
        return "figures"
    if "experiment" in rel or rel.startswith("benchmarks/"):
        return "experiments"
    return "misc"

def collate(files: list[Path]) -> list[tuple[str, Path, Path]]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, Path, Path]] = []
    for src in files:
        cat = classify(src)
        dst_dir = OUT_ROOT / cat
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        # If name collision, prefix with parent folder
        if dst.exists() and not dst.samefile(src):
            prefix = src.parent.name
            dst = dst_dir / f"{prefix}__{src.name}"
        shutil.copy2(src, dst)
        rows.append((cat, src, dst))
    return rows

def write_manifest(rows: list[tuple[str, Path, Path]]) -> None:
    manifest = OUT_ROOT / "MANIFEST.md"
    lines = ["# Collated Markdown Manifest\n"]
    by_cat: dict[str, list[tuple[Path, Path]]] = {}
    for cat, src, dst in rows:
        by_cat.setdefault(cat, []).append((src, dst))
    for cat in sorted(by_cat):
        lines.append(f"\n## {cat}\n")
        for src, dst in sorted(by_cat[cat], key=lambda x: x[1].name.lower()):
            lines.append(f"- {dst.name}  ")
            lines.append(f"  - source: {src.relative_to(REPO)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    files = find_markdown_files()
    rows = collate(files)
    write_manifest(rows)
    print(f"Collated {len(rows)} markdown files into {OUT_ROOT}")

if __name__ == "__main__":
    main()

