#!/usr/bin/env python3
"""Restore the MAD conservation-error legend location to Matplotlib's best placement."""

from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path


def function_region(source: str, function_name: str) -> tuple[int, int, str]:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {function_name}() definition; found {len(matches)}."
        )

    node = matches[0]
    if node.end_lineno is None:
        raise RuntimeError("Python 3.8 or newer is required.")

    lines = source.splitlines(keepends=True)
    first_line = min(
        [node.lineno] + [decorator.lineno for decorator in node.decorator_list]
    )
    start = sum(len(line) for line in lines[: first_line - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return start, end, source[start:end]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python patch_legend_best.py polished_error_history.py"
        )

    target = Path(sys.argv[1]).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist:\n{target}")

    source = target.read_text(encoding="utf-8")
    compile(source, str(target), "exec")

    start, end, function_text = function_region(
        source,
        "plot_saved_mad_moving_average",
    )

    old_block = '''    raw_axis.legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
    )
'''
    new_block = '''    raw_axis.legend(
        handles=handles,
        loc="best",
        fontsize=8,
    )
'''

    if new_block in function_text:
        print("No changes were necessary.")
        print("The MAD conservation-error legend already uses loc=\"best\".")
        return

    count = function_text.count(old_block)
    if count != 1:
        raise RuntimeError(
            "Expected exactly one MAD legend block with loc=\"upper left\"; "
            f"found {count}. No changes were written."
        )

    updated_function = function_text.replace(old_block, new_block, 1)
    updated = source[:start] + updated_function + source[end:]
    compile(updated, str(target), "exec")

    _, _, verified_function = function_region(
        updated,
        "plot_saved_mad_moving_average",
    )
    if new_block not in verified_function or old_block in verified_function:
        raise RuntimeError("Legend-location validation failed. No changes were written.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.name}.{timestamp}.bak")
    shutil.copy2(target, backup)

    try:
        target.write_text(updated, encoding="utf-8")
        compile(target.read_text(encoding="utf-8"), str(target), "exec")
    except Exception:
        shutil.copy2(backup, target)
        raise

    print("Patch applied successfully.")
    print(f"Updated: {target}")
    print(f"Backup:  {backup}")
    print('MAD conservation-error legend location: loc="best"')


if __name__ == "__main__":
    main()
