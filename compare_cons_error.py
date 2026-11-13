#!/usr/bin/env python3
"""Safely add the scaled raw-window normalized-MAD plot.

Usage:
    python patch_scaled_raw_mad_safe.py error_history.py
    python patch_scaled_raw_mad_safe.py error_history.py -o patched.py

This patcher avoids nested escape problems by constructing the generated plot
title with two set_title calls. It never embeds a backslash-n escape in the
inserted source code.
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "SCALED_RAW_MAD_SAFE_PATCH_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} anchor, found {count}. "
            "Use the clean code version supplied with this patcher."
        )
    return text.replace(old, new, 1)


def patch_source(text: str) -> str:
    if MARKER in text:
        raise RuntimeError("This patch is already present in the target.")

    text = replace_once(
        text,
        '    "saved_median_scaled_mad": False,\n    "rolling_log10_mad": False,',
        '    "saved_median_scaled_mad": False,\n'
        '    "scaled_raw_window_normalized_mad": True,\n'
        '    "rolling_log10_mad": False,',
        "plot toggle",
    )

    config = '''CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]

# SCALED_RAW_MAD_SAFE_PATCH_V1
SCALED_RAW_MAD_REFERENCE_ITERATIONS = 100.0
CASE_UPDATE_FREQUENCY_PATTERN = re.compile(
    r"^s[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)_u"
    r"(?P<u>[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))$",
    re.IGNORECASE,
)
'''
    text = replace_once(
        text,
        'CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]\n',
        config,
        "configuration",
    )

    text = replace_once(
        text,
        '        "saved_median_scaled_mad",\n        "rolling_log10_mad",',
        '        "saved_median_scaled_mad",\n'
        '        "scaled_raw_window_normalized_mad",\n'
        '        "rolling_log10_mad",',
        "validation key",
    )

    text = replace_once(
        text,
        '    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")\n',
        '    require_positive(\n'
        '        SCALED_RAW_MAD_REFERENCE_ITERATIONS,\n'
        '        "SCALED_RAW_MAD_REFERENCE_ITERATIONS",\n'
        '    )\n\n'
        '    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")\n',
        "configuration validation",
    )

    helpers = '''def find_case_update_frequency(path):
    """Extract u from an ancestor directory named s#_u#."""
    path = Path(path).expanduser().resolve()
    candidates = [path.name]
    candidates.extend(parent.name for parent in path.parents)
    for candidate in candidates:
        match = CASE_UPDATE_FREQUENCY_PATTERN.fullmatch(candidate)
        if match:
            update_frequency = float(match.group("u"))
            require_positive(
                update_frequency,
                "case sampling update frequency",
            )
            return update_frequency, candidate
    raise ValueError(
        "No ancestor directory named s#_u# was found for: "
        + str(path)
    )


def scaled_raw_mad_window_points(update_frequency):
    """Return floor(100/u) raw rows; u=5 gives 20 and u=20 gives 5."""
    require_positive(
        update_frequency,
        "case sampling update frequency",
    )
    return max(
        1,
        int(np.floor(
            SCALED_RAW_MAD_REFERENCE_ITERATIONS / update_frequency
        )),
    )


'''
    text = replace_once(
        text,
        'def rolling_log10_median_mad(values, window):\n',
        helpers + 'def rolling_log10_median_mad(values, window):\n',
        "numerical helpers",
    )

    # No backslash-n appears anywhere in this generated function. The title is
    # assembled with chr(10), eliminating nested string-escape interpretation.
    plotter = '''def plot_scaled_raw_window_normalized_mad_metric(
    iterations,
    raw,
    update_frequency,
    case_name,
    name,
    title,
    pdf,
    logy=False,
):
    """Plot normalized MAD over all raw rows using floor(100/u) rows."""
    iterations = np.asarray(iterations, dtype=float)
    raw = np.asarray(raw, dtype=float)
    validate_history_shape(
        raw,
        iterations,
        f"{name} raw conservation error",
    )

    window_points = scaled_raw_mad_window_points(update_frequency)
    median, _, normalized_mad, _, _ = rolling_median_spread_statistics(
        raw,
        window_points,
        ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE,
    )

    print_section(
        f"Scaled raw-window normalized MAD: {name}",
        case_name=case_name,
        update_frequency=update_frequency,
        calculated_window_points=window_points,
        raw_value_count=len(raw),
    )

    fig, raw_axis = plt.subplots(figsize=(12, 7))
    mad_axis = raw_axis.twinx()
    raw_axis.set_zorder(1)
    mad_axis.set_zorder(2)
    mad_axis.patch.set_visible(False)

    raw_artist = raw_axis.scatter(
        iterations,
        positive_or_nan(raw),
        color="0.65",
        s=9,
        alpha=0.30,
        label="Raw absolute conservation error",
        rasterized=True,
        zorder=1,
    )
    median_line, = raw_axis.plot(
        iterations,
        positive_or_nan(median),
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        label=f"Rolling error median ({window_points} raw points)",
        zorder=2,
    )

    mad_plot = positive_or_nan(normalized_mad) if logy else normalized_mad
    mad_line, = mad_axis.plot(
        iterations,
        mad_plot,
        color="tab:brown",
        linewidth=2.2,
        alpha=0.95,
        label=f"Rolling normalized MAD ({window_points} raw points)",
        zorder=4,
    )

    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size and np.all(finite_raw > 0.0):
        raw_axis.set_yscale("log")
    if logy:
        mad_axis.set_yscale("log")
    else:
        mad_axis.set_ylim(bottom=0.0)

    raw_axis.set_ylabel("Absolute conservation error")
    mad_axis.set_ylabel(
        "Normalized MAD = MAD / |median|",
        color="tab:brown",
    )
    mad_axis.tick_params(axis="y", colors="tab:brown")
    mad_axis.spines["right"].set_color("tab:brown")

    title_line_1 = f"{title}: {name}"
    title_line_2 = (
        f"Raw-data normalized MAD [{case_name}, "
        f"u={update_frequency:g}, "
        f"window=floor(100/u)={window_points} raw points]"
    )
    raw_axis.set_title(title_line_1 + chr(10) + title_line_2)

    apply_standard_axes(raw_axis)
    raw_axis.grid(False, which="both", axis="y")
    mad_axis.minorticks_on()
    mad_axis.grid(
        which="major",
        axis="y",
        linestyle="-",
        alpha=0.35,
    )
    mad_axis.grid(
        which="minor",
        axis="y",
        linestyle=":",
        alpha=0.20,
    )
    mad_axis.set_axisbelow(True)
    raw_axis.legend(
        handles=[raw_artist, median_line, mad_line],
        loc="best",
        fontsize=8,
    )
    fig.subplots_adjust(right=0.88)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


'''
    text = replace_once(
        text,
        'def plot_log10_mad_metric(\n',
        plotter + 'def plot_log10_mad_metric(\n',
        "plot function",
    )

    text = replace_once(
        text,
        '    rom_error_dir = Path(rom_error_dir)\n'
        '    iterations, values = load_error_signal_file(',
        '    rom_error_dir = Path(rom_error_dir)\n'
        '    scaled_raw_update_frequency = None\n'
        '    scaled_raw_case_name = None\n'
        '    if is_plot_enabled("scaled_raw_window_normalized_mad"):\n'
        '        (\n'
        '            scaled_raw_update_frequency,\n'
        '            scaled_raw_case_name,\n'
        '        ) = find_case_update_frequency(rom_error_dir)\n\n'
        '    iterations, values = load_error_signal_file(',
        "frequency extraction",
    )

    call = '''        if is_plot_enabled("scaled_raw_window_normalized_mad"):
            plot_scaled_raw_window_normalized_mad_metric(
                iterations=iterations,
                raw=raw,
                update_frequency=scaled_raw_update_frequency,
                case_name=scaled_raw_case_name,
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
                logy=ROLLING_NORMALIZED_MAD_LOGY,
            )

        if is_plot_enabled("rolling_log10_mad"):
            plot_log10_mad_metric('''
    text = replace_once(
        text,
        '        if is_plot_enabled("rolling_log10_mad"):\n'
        '            plot_log10_mad_metric(',
        call,
        "plot call",
    )

    text = replace_once(
        text,
        '                "saved_median_scaled_mad",\n'
        '                "rolling_log10_mad",',
        '                "saved_median_scaled_mad",\n'
        '                "scaled_raw_window_normalized_mad",\n'
        '                "rolling_log10_mad",',
        "orchestration key",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target does not exist: {target}")
    output = args.output.expanduser().resolve() if args.output else target
    original = target.read_text(encoding="utf-8")
    patched = patch_source(original)

    backup = None
    if output == target:
        backup = target.with_name(target.name + ".bak")
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(patched, encoding="utf-8")
    try:
        py_compile.compile(str(output), doraise=True)
    except Exception:
        if backup is not None:
            shutil.copy2(backup, target)
            print(f"Restored original: {target}")
        raise

    print(f"Patched: {output}")
    print('Toggle: PLOT_ENABLED["scaled_raw_window_normalized_mad"]')


if __name__ == "__main__":
    main()
