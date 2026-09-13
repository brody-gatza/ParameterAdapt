#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

OLD = '    raw_line, = raw_axis.plot(\n        iterations,\n        raw,\n        color="0.55",\n        linewidth=1,\n        alpha=0.45,\n        label="Raw error signal",\n        zorder=1,\n    )\n    handles = [raw_line]'
NEW = '    raw_artist = raw_axis.scatter(\n        iterations,\n        raw,\n        color="0.55",\n        s=9,\n        alpha=0.45,\n        label="Raw error signal",\n        rasterized=True,\n        zorder=1,\n    )\n    handles = [raw_artist]'


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Change rolling-metric raw data from a line to a scatter plot."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = args.source.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw:
        raise SystemExit("Refusing UTF-16 or source containing null bytes")

    text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    normalized = text.replace("\r\n", "\n")

    old_count = normalized.count(OLD)
    new_count = normalized.count(NEW)
    if new_count == 1 and old_count == 0:
        raise SystemExit("Raw-data scatter change is already applied")
    if old_count != 1:
        raise SystemExit(
            f"Expected raw-data line block exactly once; found {old_count}"
        )

    result = normalized.replace(OLD, NEW, 1)
    ast.parse(result)

    if "\ufffd" in result or "\x00" in result:
        raise SystemExit("Refusing corrupted patched output")

    if args.dry_run:
        print("Dry run passed: target found once and patched source parses")
        return

    with args.source.open("w", encoding="utf-8", newline="") as stream:
        stream.write(result.replace("\n", newline))

    print(f"Patched and syntax-validated: {args.source}")


if __name__ == "__main__":
    main()
