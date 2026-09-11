#!/usr/bin/env python3
"""Recursively replace ``base_url`` values in YAML files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator


NEW_BASE_URL = "https://api.inferera.com/v1"
# 修改为需要递归处理的目录；可以是绝对路径，也可以是相对当前工作目录的路径。
TARGET_DIRECTORY = Path("configs/defenses/ai4sci_llm/base_models")

# Keep the key's indentation, spacing, quote style and any inline comment intact.
BASE_URL_PATTERN = re.compile(
    r"^(?P<prefix>[ \t]*base_url[ \t]*:[ \t]*)(?P<quote>['\"]?)(?P<value>[^'\"\s#]*)(?P=quote)(?P<suffix>[ \t]*(?:#.*)?(?:\r?\n|$))$",
    re.MULTILINE,
)
YAML_SUFFIXES = {".yaml", ".yml"}


def iter_yaml_files(directory: Path) -> Iterator[Path]:
    """Yield YAML files below *directory*, in a deterministic order."""
    yield from sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in YAML_SUFFIXES
        ),
        key=lambda path: str(path),
    )


def replace_base_urls(content: str) -> tuple[str, int]:
    """Replace every ``base_url`` scalar value and return (content, count)."""

    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        if match["value"] == NEW_BASE_URL:
            return match.group(0)
        replacements += 1
        return f'{match["prefix"]}{match["quote"]}{NEW_BASE_URL}{match["quote"]}{match["suffix"]}'

    return BASE_URL_PATTERN.sub(replace, content), replacements


def main() -> int:
    directory = TARGET_DIRECTORY.expanduser()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    changed_files = 0
    replacements = 0
    for path in iter_yaml_files(directory):
        content = path.read_text(encoding="utf-8")
        updated, count = replace_base_urls(content)
        if count == 0:
            continue

        changed_files += 1
        replacements += count
        path.write_text(updated, encoding="utf-8")
        print(f"Updated: {path} ({count} base_url value(s))")

    print(f"Updated {replacements} base_url value(s) in {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
