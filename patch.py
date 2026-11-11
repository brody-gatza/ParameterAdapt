#!/usr/bin/env python3
"""
Use the Pearson-correlation y-axis ticks for horizontal gridlines on figures
that combine relative L2 error and Pearson correlation.

The patch modifies:

    plot_primitive_metrics()
    plot_interpolation_error_signals()

Result
------
- Vertical gridlines continue to use the shared iteration x-axis.
- Horizontal gridlines use the Pearson-correlation y-axis.
- Horizontal gridlines from the primary and L2-error axes are disabled.
- Major and minor correlation gridlines are drawn.
- The correlation axis remains inverted from 1 at the top to 0 at the bottom.

Usage
-----
python patch.py path/to/error_history.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PATCH_MARKER = "# CORRELATION_AXIS_GRID_PATCH_APPLIED"


def find_function_bounds(
    text: str,
    function_name: str,
) -> tuple[int, int]:
    """
    Return the character range occupied by one top-level function.
    """
    marker = f"def {function_name}("
    start = text.find(marker)

    if start < 0:
        raise RuntimeError(
            f"Could not find function {function_name!r}."
        )

    next_function = text.find("\ndef ", start + len(marker))

    if next_function < 0:
        end = len(text)
    else:
        end = next_function + 1

    return start, end


def replace_in_function(
    text: str,
    function_name: str,
    old: str,
    new: str,
    description: str,
    expected_count: int = 1,
) -> str:
    """
    Replace an exact block only inside the specified function.
    """
    start, end = find_function_bounds(text, function_name)
    function_text = text[start:end]

    count = function_text.count(old)

    if count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} occurrence(s) for {description} "
            f"inside {function_name}(), but found {count}."
        )

    function_text = function_text.replace(
        old,
        new,
        expected_count,
    )

    return text[:start] + function_text + text[end:]


def add_patch_marker(text: str) -> str:
    """
    Add the patch marker near the top of the target script.
    """
    if PATCH_MARKER in text:
        raise RuntimeError(
            "The target already contains the correlation-axis grid patch."
        )

    anchor = (
        "# Generate selected FOM/ROM comparison and "
        "conservation-error plots.\n"
    )

    count = text.count(anchor)

    if count != 1:
        raise RuntimeError(
            "Could not uniquely locate the plotting-script header. "
            f"Expected one occurrence but found {count}."
        )

    return text.replace(
        anchor,
        anchor + PATCH_MARKER + "\n",
        1,
    )


def patch_primitive_metric_grid(text: str) -> str:
    """
    Make plot_primitive_metrics() use the correlation axis for horizontal
    gridlines.
    """
    old = (
        "        error_axis.set_title(name)\n"
        "        apply_standard_axes(error_axis)\n"
        "        error_axis.legend(handles=[error_line, correlation_line], loc=\"best\")\n"
        "        save_figure(pdf, fig)\n"
    )

    new = (
        "        error_axis.set_title(name)\n"
        "        apply_standard_axes(error_axis)\n"
        "\n"
        "        # Use the Pearson-correlation axis for horizontal gridlines.\n"
        "        # Keep the vertical gridlines from the shared iteration axis.\n"
        "        error_axis.grid(False, which=\"both\", axis=\"y\")\n"
        "        correlation_axis.minorticks_on()\n"
        "        correlation_axis.grid(\n"
        "            which=\"major\",\n"
        "            axis=\"y\",\n"
        "            linestyle=\"-\",\n"
        "            alpha=0.35,\n"
        "        )\n"
        "        correlation_axis.grid(\n"
        "            which=\"minor\",\n"
        "            axis=\"y\",\n"
        "            linestyle=\":\",\n"
        "            alpha=0.20,\n"
        "        )\n"
        "        correlation_axis.set_axisbelow(True)\n"
        "\n"
        "        error_axis.legend(\n"
        "            handles=[error_line, correlation_line],\n"
        "            loc=\"best\",\n"
        "        )\n"
        "        save_figure(pdf, fig)\n"
    )

    return replace_in_function(
        text=text,
        function_name="plot_primitive_metrics",
        old=old,
        new=new,
        description="primitive L2/correlation grid configuration",
    )


def patch_interpolation_metric_grid(text: str) -> str:
    """
    Make plot_interpolation_error_signals() use the correlation axis for
    horizontal gridlines.
    """
    old = (
        "            signal_axis.set_title(f\"{title}: {field_name}\")\n"
        "            apply_standard_axes(signal_axis)\n"
        "            signal_axis.legend(\n"
        "                handles,\n"
        "                [handle.get_label() for handle in handles],\n"
        "                loc=\"best\",\n"
        "                fontsize=9,\n"
        "            )\n"
    )

    new = (
        "            signal_axis.set_title(f\"{title}: {field_name}\")\n"
        "            apply_standard_axes(signal_axis)\n"
        "\n"
        "            # Use the Pearson-correlation axis for horizontal\n"
        "            # gridlines. The signal axis retains the shared vertical\n"
        "            # iteration gridlines.\n"
        "            signal_axis.grid(False, which=\"both\", axis=\"y\")\n"
        "            l2_axis.grid(False, which=\"both\", axis=\"y\")\n"
        "            correlation_axis.minorticks_on()\n"
        "            correlation_axis.grid(\n"
        "                which=\"major\",\n"
        "                axis=\"y\",\n"
        "                linestyle=\"-\",\n"
        "                alpha=0.35,\n"
        "            )\n"
        "            correlation_axis.grid(\n"
        "                which=\"minor\",\n"
        "                axis=\"y\",\n"
        "                linestyle=\":\",\n"
        "                alpha=0.20,\n"
        "            )\n"
        "            correlation_axis.set_axisbelow(True)\n"
        "\n"
        "            signal_axis.legend(\n"
        "                handles,\n"
        "                [handle.get_label() for handle in handles],\n"
        "                loc=\"best\",\n"
        "                fontsize=9,\n"
        "            )\n"
    )

    return replace_in_function(
        text=text,
        function_name="plot_interpolation_error_signals",
        old=old,
        new=new,
        description="interpolation L2/correlation grid configuration",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "    python patch.py path/to/error_history.py"
        )

    target = Path(sys.argv[1]).expanduser().resolve()

    if not target.is_file():
        raise FileNotFoundError(
            f"Target plotting script does not exist:\n{target}"
        )

    original_text = target.read_text(encoding="utf-8")
    patched_text = original_text

    patched_text = add_patch_marker(patched_text)
    patched_text = patch_primitive_metric_grid(patched_text)
    patched_text = patch_interpolation_metric_grid(patched_text)

    if patched_text == original_text:
        raise RuntimeError(
            "The patch completed without changing the target."
        )

    # Write the backup only after all checks and replacements succeed.
    backup = target.with_name(
        target.name + ".before_correlation_grid_patch.bak"
    )

    backup_index = 1
    while backup.exists():
        backup = target.with_name(
            target.name
            + f".before_correlation_grid_patch.bak.{backup_index}"
        )
        backup_index += 1

    shutil.copy2(target, backup)
    target.write_text(patched_text, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print()
    print("Updated figure families:")
    print("    plot_primitive_metrics()")
    print("    plot_interpolation_error_signals()")
    print()
    print("Horizontal gridlines now follow Pearson-correlation ticks.")
    print("Vertical gridlines still follow the shared iteration axis.")


if __name__ == "__main__":
    main()