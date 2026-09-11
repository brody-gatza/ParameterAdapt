#!/usr/bin/env python3
"""Patch the current error-history script with log MA/EMA and RMS figures.

Usage:
    python patch_log_ma_ema_rms.txt path/to/error_history.py
"""
import argparse
import ast
import os
import shutil
import tempfile
from pathlib import Path

TARGET_FUNCTION = "plot_conservation_average_and_rms_to_pdf"
REPLACEMENT_FUNCTION = 'def plot_conservation_average_and_rms_to_pdf(\n    signal_iterations,\n    raw,\n    title,\n    field_name,\n    pdf,\n    baseline_window,\n    rms_window,\n):\n    """Append one log-baseline page and one RMS page with log raw data."""\n    signal_iterations = np.asarray(signal_iterations, dtype=float)\n    raw = np.asarray(raw, dtype=float)\n    if signal_iterations.shape != raw.shape:\n        raise ValueError(\n            "Conservation-error iterations and raw data must match."\n        )\n    if baseline_window <= 0 or rms_window <= 0:\n        raise ValueError("Baseline and RMS windows must be positive.")\n\n    moving_average = compute_moving_average_ignore_nan(raw, baseline_window)\n    exponential_average = compute_exponential_moving_average_ignore_nan(\n        raw, baseline_window\n    )\n    moving_average_rms = compute_rolling_rms_about_baseline(\n        raw, moving_average, rms_window\n    )\n    exponential_average_rms = compute_rolling_rms_about_baseline(\n        raw, exponential_average, rms_window\n    )\n\n    raw_log = np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan)\n    moving_average_log = np.where(\n        np.isfinite(moving_average) & (moving_average > 0.0),\n        moving_average,\n        np.nan,\n    )\n    exponential_average_log = np.where(\n        np.isfinite(exponential_average) & (exponential_average > 0.0),\n        exponential_average,\n        np.nan,\n    )\n\n    fig, axis = plt.subplots(figsize=(12, 7))\n    raw_artist = axis.scatter(\n        signal_iterations, raw_log, color="0.60", s=10, alpha=0.40,\n        label="ROM percent conservation error", zorder=2, rasterized=True,\n    )\n    moving_line, = axis.plot(\n        signal_iterations, moving_average_log,\n        color="tab:green", linewidth=2.6,\n        label=f"Moving average ({baseline_window})", zorder=20,\n    )\n    exponential_line, = axis.plot(\n        signal_iterations, exponential_average_log,\n        color="tab:blue", linewidth=2.6, linestyle="--",\n        label=f"Exponential moving average (span={baseline_window})", zorder=21,\n    )\n    axis.set_xlabel("Iteration")\n    axis.set_ylabel("Conservation error [%]")\n    axis.set_yscale("log")\n    axis.set_title(\n        f"{title}: {field_name}\\nMoving and exponential moving averages"\n    )\n    axis.minorticks_on()\n    axis.grid(which="major", axis="both", linestyle="-", alpha=0.35)\n    axis.grid(which="minor", axis="both", linestyle=":", alpha=0.20)\n    axis.legend(handles=[raw_artist, moving_line, exponential_line], loc="best")\n    fig.tight_layout()\n    pdf.savefig(fig, bbox_inches="tight", dpi=150)\n    plt.close(fig)\n\n    fig, rms_axis = plt.subplots(figsize=(12, 7))\n    error_axis = rms_axis.twinx()\n    moving_rms_line, = rms_axis.plot(\n        signal_iterations, moving_average_rms, color="tab:green",\n        linewidth=2.4,\n        label=f"RMS about moving-average baseline ({rms_window})",\n        zorder=20,\n    )\n    exponential_rms_line, = rms_axis.plot(\n        signal_iterations, exponential_average_rms, color="tab:blue",\n        linewidth=2.4, linestyle="--",\n        label=f"RMS about EMA baseline ({rms_window})",\n        zorder=21,\n    )\n    raw_error_artist = error_axis.scatter(\n        signal_iterations, raw_log, color="0.55", s=9, alpha=0.30,\n        label="ROM percent conservation error", zorder=2, rasterized=True,\n    )\n    rms_axis.set_xlabel("Iteration")\n    rms_axis.set_ylabel("RMS fluctuation magnitude [percentage points]")\n    error_axis.set_ylabel(\n        "Conservation error [%]", color="0.35"\n    )\n    error_axis.tick_params(axis="y", labelcolor="0.35")\n    error_axis.set_yscale("log")\n    rms_axis.set_title(\n        f"{title}: {field_name}\\n"\n        f"Baseline-relative rolling RMS [baseline={baseline_window}, "\n        f"RMS={rms_window}]"\n    )\n    rms_axis.minorticks_on()\n    rms_axis.grid(which="major", axis="both", linestyle="-", alpha=0.35)\n    rms_axis.grid(which="minor", axis="x", linestyle=":", alpha=0.20)\n    rms_axis.legend(\n        handles=[moving_rms_line, exponential_rms_line, raw_error_artist],\n        loc="best",\n    )\n    fig.tight_layout()\n    pdf.savefig(fig, bbox_inches="tight", dpi=150)\n    plt.close(fig)\n\n    print(\n        f"Added conservation averages and RMS pages: {field_name}; "\n        f"baseline={baseline_window}; RMS={rms_window}", flush=True,\n    )\n\n\n'


def find_unique_top_level_function(tree, function_name):
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level {function_name} function; "
            f"found {len(matches)}. The target file was not modified."
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Use log conservation-error axes and 1000-sample MA/EMA windows."
    )
    parser.add_argument("target", help="Python source file to patch")
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    text = original.replace("\r\n", "\n")
    tree = ast.parse(text, filename=str(target))

    function_node = find_unique_top_level_function(tree, TARGET_FUNCTION)
    if function_node.col_offset != 0:
        raise RuntimeError("Target function is not top-level; no changes made.")

    lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in lines[:function_node.lineno - 1])
    end = sum(len(line) for line in lines[:function_node.end_lineno])
    old_function = text[start:end]

    required_markers = (
        "moving_average = compute_moving_average_ignore_nan(raw, baseline_window)",
        "exponential_average = compute_exponential_moving_average_ignore_nan(",
        "moving_average_rms = compute_rolling_rms_about_baseline(",
        "exponential_average_rms = compute_rolling_rms_about_baseline(",
    )
    missing = [marker for marker in required_markers if marker not in old_function]
    if missing:
        raise RuntimeError(
            "The named function does not match the expected current version. "
            f"Missing marker: {missing[0]!r}. No changes were made."
        )

    modified = text[:start] + REPLACEMENT_FUNCTION + text[end:]

    old_call = "            baseline_window=long_window,\n"
    call_count = modified.count(old_call)
    if call_count != 1:
        raise RuntimeError(
            "Could not uniquely locate the conservation figure baseline-window "
            f"argument. Expected 1 match, found {call_count}. No changes made."
        )
    modified = modified.replace(
        old_call, "            baseline_window=1000,\n", 1
    )

    new_tree = ast.parse(modified, filename=str(target))
    find_unique_top_level_function(new_tree, TARGET_FUNCTION)
    compile(modified, str(target), "exec")

    backup = target.with_suffix(target.suffix + ".bak")
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}\n"
            "Move or remove it before applying this patch."
        )
    shutil.copy2(target, backup)

    output = modified.replace("\n", newline)
    fd, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(output)
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("MA window:  1000")
    print("EMA span:   1000")
    print("Syntax validation passed before replacement.")


if __name__ == "__main__":
    main()
