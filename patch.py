#!/usr/bin/env python3
"""
Update the conservation-error plotting script to load the short-MAD moving
average directly from error_cons_mad.txt instead of recomputing it during
post-processing.

Expected short-MAD file layout
------------------------------
error_cons_mad.txt must contain:

    iteration,
    center[0:N],
    MAD[0:N],
    moving_average[0:N]

where N is the number of conservative variables.

For four conservative variables, the complete row therefore contains:

    1 iteration column
    4 center columns
    4 MAD columns
    4 moving-average columns

The loader sees 12 value columns after load_error_signal_file() removes the
iteration column.

The long-MAD file remains unchanged:

    iteration,
    center[0:N],
    MAD[0:N]

Usage
-----
python patch_use_saved_mad_moving_average.py error_history.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PATCH_MARKER = "# SAVED_MAD_MOVING_AVERAGE_LOADING_PATCH_APPLIED"


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

    next_function = text.find(
        "\ndef ",
        start + len(marker),
    )

    if next_function < 0:
        end = len(text)
    else:
        end = next_function + 1

    return start, end


def replace_exact(
    text: str,
    old: str,
    new: str,
    description: str,
    expected_count: int = 1,
) -> str:
    """
    Replace an exact block only when the expected number of matches is found.
    """
    count = text.count(old)

    if count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} occurrence(s) for {description}, "
            f"but found {count}."
        )

    return text.replace(
        old,
        new,
        expected_count,
    )


def replace_in_function(
    text: str,
    function_name: str,
    old: str,
    new: str,
    description: str,
    expected_count: int = 1,
) -> str:
    """
    Replace an exact block only within one top-level function.
    """
    start, end = find_function_bounds(
        text,
        function_name,
    )

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

    return (
        text[:start]
        + function_text
        + text[end:]
    )


def replace_whole_function(
    text: str,
    function_name: str,
    new_function: str,
) -> str:
    """
    Replace an entire top-level function.
    """
    start, end = find_function_bounds(
        text,
        function_name,
    )

    return (
        text[:start]
        + new_function.rstrip()
        + "\n\n\n"
        + text[end:]
    )


def add_patch_marker(text: str) -> str:
    if PATCH_MARKER in text:
        raise RuntimeError(
            "The target already contains the saved-MAD moving-average "
            "loading patch."
        )

    marker_anchor = (
        "# SAVED_SHORT_MAD_MOVING_AVERAGE_PATCH_APPLIED\n"
    )

    return replace_exact(
        text,
        marker_anchor,
        marker_anchor + PATCH_MARKER + "\n",
        "patch marker insertion",
    )


def update_configuration_comment(text: str) -> str:
    old = (
        "# Number of stored short-MAD samples used by the moving-average\n"
        "# curve on the saved short-MAD moving-average figures.\n"
        "SHORT_MAD_MOVING_AVERAGE_WINDOW = 1000\n"
    )

    new = (
        "# Window used when the short-MAD moving average was calculated during\n"
        "# data collection. This value is used only for plot labels and does\n"
        "# not cause the moving average to be recomputed during post-processing.\n"
        "SHORT_MAD_MOVING_AVERAGE_WINDOW = 1000\n"
    )

    return replace_exact(
        text,
        old,
        new,
        "moving-average configuration comment",
    )


def update_saved_mad_loader(text: str) -> str:
    """
    Replace the nested load_saved_mad_history() helper.

    The short file must contain 3N columns:
        medians, MAD values, moving-average values

    The long file continues to contain 2N columns:
        medians, MAD values
    """
    old = '''    def load_saved_mad_history(filename):
        mad_iterations, mad_columns = load_error_signal_file(
            rom_error_dir / filename,
            active_range,
            delimiter,
        )

        number_of_variables = values.shape[1]
        expected_columns = 2 * number_of_variables

        if mad_columns.shape[1] != expected_columns:
            raise ValueError(
                f"Expected {filename} to contain "
                f"{expected_columns} value columns: "
                f"{number_of_variables} medians followed by "
                f"{number_of_variables} normalized MAD values; received "
                f"{mad_columns.shape[1]}."
            )

        if iterations.shape != mad_iterations.shape or not np.allclose(
            iterations,
            mad_iterations,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Raw absolute conservation-error and "
                f"{filename} are not aligned."
            )

        medians = mad_columns[:, :number_of_variables]
        normalized_mads = mad_columns[:, number_of_variables:]

        return mad_iterations, medians, normalized_mads
'''

    new = '''    def load_saved_mad_history(
        filename,
        includes_moving_average=False,
    ):
        """
        Load a saved MAD history.

        Short-file layout when includes_moving_average is True:

            medians[0:N],
            normalized_mads[0:N],
            moving_averages[0:N]

        Long-file layout when includes_moving_average is False:

            medians[0:N],
            normalized_mads[0:N]
        """
        mad_iterations, mad_columns = load_error_signal_file(
            rom_error_dir / filename,
            active_range,
            delimiter,
        )

        number_of_variables = values.shape[1]
        number_of_groups = (
            3
            if includes_moving_average
            else 2
        )
        expected_columns = (
            number_of_groups
            * number_of_variables
        )

        if mad_columns.shape[1] != expected_columns:
            if includes_moving_average:
                expected_description = (
                    f"{number_of_variables} medians, followed by "
                    f"{number_of_variables} normalized MAD values, "
                    f"followed by {number_of_variables} saved "
                    "moving-average values"
                )
            else:
                expected_description = (
                    f"{number_of_variables} medians followed by "
                    f"{number_of_variables} normalized MAD values"
                )

            raise ValueError(
                f"Expected {filename} to contain "
                f"{expected_columns} value columns: "
                f"{expected_description}; received "
                f"{mad_columns.shape[1]}."
            )

        if (
            iterations.shape != mad_iterations.shape
            or not np.allclose(
                iterations,
                mad_iterations,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "Raw absolute conservation-error and "
                f"{filename} are not aligned."
            )

        median_stop = number_of_variables
        mad_stop = 2 * number_of_variables

        medians = mad_columns[
            :,
            :median_stop,
        ]
        normalized_mads = mad_columns[
            :,
            median_stop:mad_stop,
        ]

        moving_averages = None

        if includes_moving_average:
            moving_averages = mad_columns[
                :,
                mad_stop:3 * number_of_variables,
            ]

        return (
            mad_iterations,
            medians,
            normalized_mads,
            moving_averages,
        )
'''

    return replace_in_function(
        text,
        "plot_rom_conservation_errors",
        old,
        new,
        "saved-MAD loader replacement",
    )


def add_saved_average_initialization(text: str) -> str:
    old = '''    saved_mad_iterations = None
    saved_median = None
    saved_normalized_mad = None

    saved_mad_long_iterations = None
'''

    new = '''    saved_mad_iterations = None
    saved_median = None
    saved_normalized_mad = None
    saved_normalized_mad_average = None

    saved_mad_long_iterations = None
'''

    return replace_in_function(
        text,
        "plot_rom_conservation_errors",
        old,
        new,
        "saved moving-average initialization",
    )


def update_short_file_loading(text: str) -> str:
    old = '''        (
            saved_mad_iterations,
            saved_median,
            saved_normalized_mad,
        ) = load_saved_mad_history("error_cons_mad.txt")
'''

    new = '''        (
            saved_mad_iterations,
            saved_median,
            saved_normalized_mad,
            saved_normalized_mad_average,
        ) = load_saved_mad_history(
            "error_cons_mad.txt",
            includes_moving_average=True,
        )
'''

    return replace_in_function(
        text,
        "plot_rom_conservation_errors",
        old,
        new,
        "short saved-MAD loading update",
    )


def update_long_file_loading(text: str) -> str:
    old = '''        (
            saved_mad_long_iterations,
            saved_median_long,
            saved_normalized_mad_long,
        ) = load_saved_mad_history("error_cons_mad_long.txt")
'''

    new = '''        (
            saved_mad_long_iterations,
            saved_median_long,
            saved_normalized_mad_long,
            _,
        ) = load_saved_mad_history(
            "error_cons_mad_long.txt",
            includes_moving_average=False,
        )
'''

    return replace_in_function(
        text,
        "plot_rom_conservation_errors",
        old,
        new,
        "long saved-MAD loading update",
    )


def replace_plotting_function(text: str) -> str:
    new_function = r'''def plot_saved_short_mad_moving_average(
    iterations,
    raw,
    short_median,
    short_mad,
    short_mad_average,
    name,
    title,
    pdf,
):
    """
    Plot the saved short median, short normalized MAD, and saved MAD average.

    Left logarithmic axis:
        raw absolute conservation error
        saved short-window error median

    Right linear axis:
        saved short-window normalized MAD
        saved moving average of the short normalized MAD

    The moving-average values are read directly from error_cons_mad.txt.
    No moving-average calculation is performed during post-processing.
    """
    iterations = np.asarray(
        iterations,
        dtype=float,
    )
    raw = np.asarray(
        raw,
        dtype=float,
    )
    short_median = np.asarray(
        short_median,
        dtype=float,
    )
    short_mad = np.asarray(
        short_mad,
        dtype=float,
    )
    short_mad_average = np.asarray(
        short_mad_average,
        dtype=float,
    )

    validate_history_shape(
        raw,
        iterations,
        f"{name} raw conservation error",
    )
    validate_history_shape(
        short_median,
        iterations,
        f"{name} short saved median",
    )
    validate_history_shape(
        short_mad,
        iterations,
        f"{name} short saved MAD",
    )
    validate_history_shape(
        short_mad_average,
        iterations,
        f"{name} saved short-MAD moving average",
    )

    fig, raw_axis = plt.subplots(
        figsize=(12, 7)
    )
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

    short_median_line, = raw_axis.plot(
        iterations,
        positive_or_nan(short_median),
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        label="Short-window error median (100)",
        zorder=2,
    )

    short_mad_line, = mad_axis.plot(
        iterations,
        short_mad,
        color="tab:brown",
        linewidth=1.4,
        alpha=0.55,
        label="Short-window normalized MAD (100)",
        zorder=4,
    )

    average_line, = mad_axis.plot(
        iterations,
        short_mad_average,
        color="tab:orange",
        linewidth=2.8,
        alpha=1.0,
        label=(
            "Saved moving average of short normalized MAD "
            f"({SHORT_MAD_MOVING_AVERAGE_WINDOW})"
        ),
        zorder=5,
    )

    raw_axis.set_yscale("log")
    raw_axis.set_ylabel(
        "Absolute conservation error"
    )

    mad_axis_color = "tab:orange"

    mad_axis.set_ylabel(
        "Short normalized MAD and saved moving average",
        color=mad_axis_color,
    )
    mad_axis.tick_params(
        axis="y",
        colors=mad_axis_color,
    )
    mad_axis.spines["right"].set_color(
        mad_axis_color
    )
    mad_axis.set_ylim(bottom=0.0)

    raw_axis.set_title(
        f"{title}: {name}\n"
        "Short normalized MAD and saved moving-average baseline"
    )

    apply_standard_axes(raw_axis)

    # Keep vertical iteration gridlines from the shared x-axis, but use the
    # right MAD axis for horizontal gridlines.
    raw_axis.grid(
        False,
        which="both",
        axis="y",
    )

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

    handles = [
        raw_artist,
        short_median_line,
        short_mad_line,
        average_line,
    ]

    raw_axis.legend(
        handles=handles,
        loc="best",
        fontsize=8,
    )

    finite_short_mad = short_mad[
        np.isfinite(short_mad)
    ]
    finite_average = short_mad_average[
        np.isfinite(short_mad_average)
    ]

    if (
        finite_short_mad.size
        and finite_average.size
    ):
        print_section(
            f"Saved short-MAD moving average: {name}",
            moving_average_window=(
                SHORT_MAD_MOVING_AVERAGE_WINDOW
            ),
            median_short_mad=float(
                np.median(finite_short_mad)
            ),
            maximum_short_mad=float(
                np.max(finite_short_mad)
            ),
            median_saved_short_mad_average=float(
                np.median(finite_average)
            ),
            maximum_saved_short_mad_average=float(
                np.max(finite_average)
            ),
        )

    fig.subplots_adjust(right=0.88)

    pdf.savefig(
        fig,
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)'''

    return replace_whole_function(
        text,
        "plot_saved_short_mad_moving_average",
        new_function,
    )


def update_plot_call(text: str) -> str:
    old = '''        if is_plot_enabled("saved_short_mad_moving_average"):
            plot_saved_short_mad_moving_average(
                iterations=saved_mad_iterations,
                raw=raw,
                short_median=saved_median[:, i],
                short_mad=saved_normalized_mad[:, i],
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
            )
'''

    new = '''        if is_plot_enabled("saved_short_mad_moving_average"):
            plot_saved_short_mad_moving_average(
                iterations=saved_mad_iterations,
                raw=raw,
                short_median=saved_median[:, i],
                short_mad=saved_normalized_mad[:, i],
                short_mad_average=(
                    saved_normalized_mad_average[:, i]
                ),
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
            )
'''

    return replace_in_function(
        text,
        "plot_rom_conservation_errors",
        old,
        new,
        "saved moving-average plot call",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "    python patch_use_saved_mad_moving_average.py "
            "path/to/error_history.py"
        )

    target = Path(
        sys.argv[1]
    ).expanduser().resolve()

    if not target.is_file():
        raise FileNotFoundError(
            f"Target plotting script does not exist:\n{target}"
        )

    original_text = target.read_text(
        encoding="utf-8"
    )

    patched_text = original_text

    patched_text = add_patch_marker(
        patched_text
    )
    patched_text = update_configuration_comment(
        patched_text
    )
    patched_text = update_saved_mad_loader(
        patched_text
    )
    patched_text = add_saved_average_initialization(
        patched_text
    )
    patched_text = update_short_file_loading(
        patched_text
    )
    patched_text = update_long_file_loading(
        patched_text
    )
    patched_text = replace_plotting_function(
        patched_text
    )
    patched_text = update_plot_call(
        patched_text
    )

    if patched_text == original_text:
        raise RuntimeError(
            "The patch completed without modifying the target."
        )

    # Create the backup only after all patch operations succeed.
    backup = target.with_name(
        target.name
        + ".before_saved_mad_ma_loading_patch.bak"
    )

    backup_number = 1

    while backup.exists():
        backup = target.with_name(
            target.name
            + ".before_saved_mad_ma_loading_patch"
            + f".bak.{backup_number}"
        )
        backup_number += 1

    shutil.copy2(
        target,
        backup,
    )

    target.write_text(
        patched_text,
        encoding="utf-8",
    )

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print()
    print("Updated short-file layout:")
    print("    medians[0:N]")
    print("    normalized_mads[0:N]")
    print("    moving_averages[0:N]")
    print()
    print(
        "The moving-average curve is now loaded from "
        "error_cons_mad.txt."
    )
    print()
    print("Validate the modified script with:")
    print(
        f"    python -m py_compile {target.name}"
    )


if __name__ == "__main__":
    main()