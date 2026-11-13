#!/usr/bin/env python3
"""Safely add an iteration-scaled raw normalized-MAD figure.

The generated figure computes a causal normalized MAD over the complete raw
error_cons.txt history, but converts the requested iteration interval into a
number of raw rows using the case update frequency encoded in an ancestor
folder named s#_u#.

For the default 100-iteration interval:
    u=5  -> floor(100/5)  = 20 raw rows
    u=20 -> floor(100/20) = 5 raw rows

Usage:
    python patch_iteration_scaled_mad.py target.py
    python patch_iteration_scaled_mad.py target.py -o patched.py
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


MARKER = "ITERATION_SCALED_RAW_MAD_PATCH_V1"
TOGGLE = "iteration_scaled_raw_normalized_mad"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} anchor, found {count}. "
            "Apply this patcher to the unpatched code supplied with the request."
        )
    return text.replace(old, new, 1)


def patch_source(text: str) -> str:
    if MARKER in text:
        raise RuntimeError("This patch is already present in the target.")

    # Add the independent plot toggle.
    text = replace_once(
        text,
        '    "saved_median_scaled_mad": False,\n    "rolling_log10_mad": False,',
        '    "saved_median_scaled_mad": False,\n'
        f'    "{TOGGLE}": True,\n'
        '    "rolling_log10_mad": False,',
        "PLOT_ENABLED toggle",
    )

    # Add the requested physical iteration interval and the directory-name
    # parser. The existing re import is reused.
    configuration = '''CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]

# ITERATION_SCALED_RAW_MAD_PATCH_V1
ITERATION_SCALED_MAD_WINDOW = 100.0
CASE_UPDATE_FREQUENCY_PATTERN = re.compile(
    r"^s[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)_u"
    r"(?P<u>[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))$",
    re.IGNORECASE,
)
'''
    text = replace_once(
        text,
        'CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]\n',
        configuration,
        "configuration insertion",
    )

    # Add the toggle to exact-key validation.
    text = replace_once(
        text,
        '        "saved_median_scaled_mad",\n        "rolling_log10_mad",',
        '        "saved_median_scaled_mad",\n'
        f'        "{TOGGLE}",\n'
        '        "rolling_log10_mad",',
        "validation key",
    )

    # Validate the new iteration interval.
    text = replace_once(
        text,
        '    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")\n',
        '    require_positive(\n'
        '        ITERATION_SCALED_MAD_WINDOW,\n'
        '        "ITERATION_SCALED_MAD_WINDOW",\n'
        '    )\n\n'
        '    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")\n',
        "configuration validation",
    )

    helpers = '''def find_case_update_frequency(path):
    """Extract update frequency u from this path or an s#_u# ancestor."""
    path = Path(path).expanduser().resolve()
    candidates = [path.name]
    candidates.extend(parent.name for parent in path.parents)

    for candidate in candidates:
        match = CASE_UPDATE_FREQUENCY_PATTERN.fullmatch(candidate)
        if match:
            update_frequency = float(match.group("u"))
            require_positive(
                update_frequency,
                "case update frequency",
            )
            return update_frequency, candidate

    raise ValueError(
        "No path component matching s#_u# was found for: " + str(path)
    )


def iteration_scaled_mad_window_points(update_frequency):
    """Convert the configured iteration interval to a raw-row window."""
    require_positive(update_frequency, "case update frequency")
    return max(
        1,
        int(np.floor(
            ITERATION_SCALED_MAD_WINDOW / update_frequency
        )),
    )


'''
    text = replace_once(
        text,
        'def rolling_log10_median_mad(values, window):\n',
        helpers + 'def rolling_log10_median_mad(values, window):\n',
        "numerical helper insertion",
    )

    # This follows the existing gMAD/MAD figure styling: raw scatter and
    # median on a logarithmic left axis, normalized MAD on a brown right axis,
    # right-axis horizontal grids, and the standard 12 x 7 layout.
    plotter = '''def plot_iteration_scaled_raw_normalized_mad_metric(
    iterations,
    raw,
    update_frequency,
    case_name,
    name,
    title,
    pdf,
    logy=False,
):
    """Plot normalized MAD using a window defined in iteration units."""
    iterations = np.asarray(iterations, dtype=float)
    raw = np.asarray(raw, dtype=float)

    validate_history_shape(
        raw,
        iterations,
        f"{name} raw conservation error",
    )

    window_points = iteration_scaled_mad_window_points(update_frequency)
    median, _, normalized_mad, _, _ = rolling_median_spread_statistics(
        raw,
        window_points,
        ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE,
    )

    print_section(
        f"Iteration-scaled raw normalized MAD: {name}",
        case_name=case_name,
        update_frequency=update_frequency,
        requested_iteration_window=ITERATION_SCALED_MAD_WINDOW,
        calculated_raw_window_points=window_points,
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
        alpha=0.35,
        label="Raw absolute conservation error",
        rasterized=True,
        zorder=1,
    )

    median_line, = raw_axis.plot(
        iterations,
        positive_or_nan(median),
        color="tab:green",
        linewidth=1.7,
        linestyle="--",
        alpha=0.90,
        label=(
            "Iteration-scaled absolute-error median "
            f"({window_points} raw points)"
        ),
        zorder=2,
    )

    mad_plot = positive_or_nan(normalized_mad) if logy else normalized_mad
    mad_line, = mad_axis.plot(
        iterations,
        mad_plot,
        color="tab:brown",
        linewidth=1.8,
        alpha=0.90,
        label=(
            "Iteration-scaled normalized MAD "
            f"({window_points} raw points)"
        ),
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

    mad_axis_color = "tab:brown"
    mad_axis.set_ylabel(
        "Normalized MAD = MAD / |median|",
        color=mad_axis_color,
    )
    mad_axis.tick_params(axis="y", colors=mad_axis_color)
    mad_axis.spines["right"].set_color(mad_axis_color)

    title_line_1 = f"{title}: {name}"
    title_line_2 = (
        "Iteration-scaled raw normalized MAD "
        f"[{case_name}, u={update_frequency:g}, "
        f"window=floor({ITERATION_SCALED_MAD_WINDOW:g}/u)="
        f"{window_points} raw points]"
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
        fontsize=9,
    )

    fig.subplots_adjust(right=0.88)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


'''
    text = replace_once(
        text,
        'def plot_log10_mad_metric(\n',
        plotter + 'def plot_log10_mad_metric(\n',
        "plot-function insertion",
    )

    # Only parse s#_u# when this new figure is enabled.
    text = replace_once(
        text,
        '    rom_error_dir = Path(rom_error_dir)\n'
        '    iterations, values = load_error_signal_file(',
        '    rom_error_dir = Path(rom_error_dir)\n'
        '    iteration_scaled_update_frequency = None\n'
        '    iteration_scaled_case_name = None\n'
        f'    if is_plot_enabled("{TOGGLE}"):\n'
        '        (\n'
        '            iteration_scaled_update_frequency,\n'
        '            iteration_scaled_case_name,\n'
        '        ) = find_case_update_frequency(rom_error_dir)\n\n'
        '    iterations, values = load_error_signal_file(',
        "case-frequency extraction",
    )

    # Add one new PDF page for every conservative variable.
    call = f'''        if is_plot_enabled("{TOGGLE}"):
            plot_iteration_scaled_raw_normalized_mad_metric(
                iterations=iterations,
                raw=raw,
                update_frequency=iteration_scaled_update_frequency,
                case_name=iteration_scaled_case_name,
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
        "per-variable plot call",
    )

    # Ensure the ROM conservation-error routine is orchestrated even when this
    # is the only enabled conservation-error figure.
    text = replace_once(
        text,
        '                "saved_median_scaled_mad",\n'
        '                "rolling_log10_mad",',
        '                "saved_median_scaled_mad",\n'
        f'                "{TOGGLE}",\n'
        '                "rolling_log10_mad",',
        "orchestration key",
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add an iteration-scaled raw normalized-MAD figure."
    )
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
        else:
            output.unlink(missing_ok=True)
            print(f"Removed invalid output: {output}")
        raise

    print(f"Patched: {output}")
    print(f'Toggle: PLOT_ENABLED["{TOGGLE}"]')
    print(
        "Window: floor(ITERATION_SCALED_MAD_WINDOW / "
        "case update frequency) raw rows"
    )


if __name__ == "__main__":
    main()
