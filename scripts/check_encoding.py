#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kyunghoon Gwak <hibouaile04@gmail.com>
"""Fail if any tracked text file contains U+FFFD or known mojibake remnants.

Background: 1c43081 (2026-04-28) silently re-encoded 20 source files via
cp949 round-trip on a Korean Windows host because a batch SPDX-header
script omitted encoding='utf-8' from its file writes. See PR #1.

This script is a CI guard against that class of bug. Run it locally with
    python scripts/check_encoding.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_EXTS = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".cfg", ".ini"}

MOJIBAKE_RX = re.compile(r"�")


def tracked_files() -> list[Path]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True, encoding="utf-8"
    )
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def main() -> int:
    bad: list[tuple[Path, int, str]] = []
    for f in tracked_files():
        if f.suffix.lower() not in TEXT_EXTS:
            continue
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            bad.append((f, 0, f"file is not valid UTF-8: {e}"))
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if MOJIBAKE_RX.search(line):
                bad.append((f, i, line.strip()[:120]))

    if bad:
        print(f"Encoding check FAILED: {len(bad)} issue(s)\n", file=sys.stderr)
        for f, lineno, msg in bad:
            rel = f.relative_to(REPO_ROOT)
            print(f"  {rel}:{lineno}: {msg}", file=sys.stderr)
        print(
            "\nFix: re-save the file as UTF-8. Common cause is a script that\n"
            "wrote files without encoding='utf-8' on a non-UTF-8 locale.",
            file=sys.stderr,
        )
        return 1

    print(f"Encoding check OK ({sum(1 for _ in tracked_files())} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
