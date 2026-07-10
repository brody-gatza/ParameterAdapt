#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# USER SETTINGS
# ============================================================

DELIMITER = ","
SKIPROWS = 50
SHOW_GRID = True
SHOW_FIGURES = False

# Moving average settings
PLOT_MOVING_AVERAGE = True
MOVING_AVERAGE_WINDOWS = [250]

# Slope settings
CALCULATE_SLOPE = True
PLOT_SLOPE_OF_RAW = True
PLOT_SLOPE_OF_MOVING_AVERAGE = True
PLOT_SLOPE_MAGNITUDE = True
PLOT_MOVING_AVERAGE_OF_SLOPE = True
PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY = True
PLOT_POSITIVE_RAW_SLOPES_ONLY = True

# Cumulative slope settings
PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE = True
PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM = True

# Usually keep slope and cumulative-slope plots linear
SLOPE_LOGY = False
SLOPE_MAGNITUDE_LOGY = False
CUMULATIVE_SLOPE_LOGY = False

# Y-axis limit settings
#
# This replaces the old percentile-only y-limit behavior.
# The goal is:
#   - show the main body of the data
#   - only cut off values when they are extreme jumps several orders
#     of magnitude larger than the rest of the plotted data
APPLY_OUTLIER_AWARE_Y_LIMITS = True

# Percentiles used to estimate the normal/main data range.
# These are intentionally wide so ordinary data is not cut off.
MAIN_DATA_PERCENTILES = (0.1, 99.9)

# If an endpoint is more than this many orders of magnitude larger
# than the main data scale, it may be clipped.
#
# 3.0 means roughly 1000x larger.
# 4.0 means roughly 10000x larger.
EXTREME_OUTLIER_ORDERS_OF_MAGNITUDE = 3.0

# Padding added around visible y-data
Y_LIMIT_PADDING_FRACTION = 0.08

# Fixed raw-data colors assigned to files in the order they appear
FILE_COLORS = [
    "blue",
    "orange",
    "green",
    "purple",
]

# Fixed moving-average colors assigned to files in the order they appear
MOVING_AVERAGE_COLORS = [
    "cyan",
    "red",
    "lime",
    "magenta",
]

SAVE_DIR = "figures"
PDF_OUTPUT_FILE = "all_comparisons.pdf"


# ============================================================
# FIGURE GROUPS
# ============================================================

FIGURE_GROUPS = [
    {
        "title": "Primitive Variables - Max",
        "filenames": [
            "prim_interp_max.txt",
            "prim_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "prim_max_comparison.png",
    },
    {
        "title": "Primitive Variables - Avg",
        "filenames": [
            "prim_interp_avg.txt",
            "prim_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "prim_avg_comparison.png",
    },
    {
        "title": "Primitive Variables - Min",
        "filenames": [
            "prim_interp_min.txt",
            "prim_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
        "save_plot": True,
        "output_file": "prim_min_comparison.png",
    },

    {
        "title": "Conservative Variables - Max",
        "filenames": [
            "cons_interp_max.txt",
            "cons_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "cons_max_comparison.png",
    },
    {
        "title": "Conservative Variables - Avg",
        "filenames": [
            "cons_interp_avg.txt",
            "cons_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "cons_avg_comparison.png",
    },
    {
        "title": "Conservative Variables - Min",
        "filenames": [
            "cons_interp_min.txt",
            "cons_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
        "save_plot": True,
        "output_file": "cons_min_comparison.png",
    },

    {
        "title": "Gradient Primitive Variables - Max",
        "filenames": [
            "grad_prim_interp_max.txt",
            "grad_prim_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "grad_prim_max_comparison.png",
    },
    {
        "title": "Gradient Primitive Variables - Avg",
        "filenames": [
            "grad_prim_interp_avg.txt",
            "grad_prim_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "grad_prim_avg_comparison.png",
    },
    {
        "title": "Gradient Primitive Variables - Min",
        "filenames": [
            "grad_prim_interp_min.txt",
            "grad_prim_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
        "save_plot": True,
        "output_file": "grad_prim_min_comparison.png",
    },

    {
        "title": "Gradient Conservative Variables - Max",
        "filenames": [
            "grad_cons_interp_max.txt",
            "grad_cons_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "grad_cons_max_comparison.png",
    },
    {
        "title": "Gradient Conservative Variables - Avg",
        "filenames": [
            "grad_cons_interp_avg.txt",
            "grad_cons_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
        "save_plot": True,
        "output_file": "grad_cons_avg_comparison.png",
    },
    {
        "title": "Gradient Conservative Variables - Min",
        "filenames": [
            "grad_cons_interp_min.txt",
            "grad_cons_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
        "save_plot": True,
        "output_file": "grad_cons_min_comparison.png",
    },
]


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(filename, delimiter=",", skiprows=0):
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)
    iterations = data[:, 0]
    variables = data[:, 1:]
    return iterations, variables


def compute_moving_average(x, y, window):
    kernel = np.ones(window) / window
    y_avg = np.convolve(y, kernel, mode="valid")

    start = window // 2
    end = start + len(y_avg)
    x_avg = x[start:end]

    return x_avg, y_avg


def compute_moving_average_ignore_nan(x, y, window):
    valid = np.isfinite(y)

    y_clean = np.where(valid, y, 0.0)
    valid_count = np.convolve(valid.astype(float), np.ones(window), mode="valid")
    y_sum = np.convolve(y_clean, np.ones(window), mode="valid")

    with np.errstate(divide="ignore", invalid="ignore"):
        y_avg = y_sum / valid_count

    y_avg[valid_count == 0] = np.nan

    start = window // 2
    end = start + len(y_avg)
    x_avg = x[start:end]

    return x_avg, y_avg

def compute_running_average(x, y):
    """
    Compute the running/cumulative average of a data array.

    For each point i, this computes the average of all values from
    the start of the array through the current point:

        running_average[i] = mean(y[0:i+1])

    Example:
        y = [1, 2, 3, 4, 5]

        running_average = [
            1,
            1.5,
            2,
            2.5,
            3,
        ]

    This returns an array with the same length as the original x and y.
    """

    y = np.asarray(y, dtype=float)

    cumulative_sum = np.cumsum(y)
    sample_count = np.arange(1, len(y) + 1)

    running_average = cumulative_sum / sample_count

    return x, running_average


def compute_slope(x, y):
    return np.gradient(y, x)


def compute_cumulative_sum(y):
    return np.cumsum(y)


# ============================================================
# FILENAME HELPERS
# ============================================================

def add_filename_suffix(filename, suffix):
    filename = Path(filename)
    return str(filename.with_name(f"{filename.stem}{suffix}{filename.suffix}"))


def make_slope_output_file(
    output_file,
    slope_type,
    output_suffix="",
    plot_magnitude=False,
):
    output_file = Path(output_file)

    if plot_magnitude:
        slope_label = f"{slope_type}_slope_magnitude"
    else:
        slope_label = f"{slope_type}_slope"

    return str(
        output_file.with_name(
            f"{output_file.stem}_{slope_label}{output_suffix}{output_file.suffix}"
        )
    )


def make_raw_slope_moving_average_output_file(output_file, output_suffix=""):
    output_file = Path(output_file)

    return str(
        output_file.with_name(
            f"{output_file.stem}_raw_slope_moving_average_only"
            f"{output_suffix}{output_file.suffix}"
        )
    )


def make_positive_raw_slope_output_file(output_file, output_suffix=""):
    output_file = Path(output_file)

    return str(
        output_file.with_name(
            f"{output_file.stem}_positive_raw_slope_only"
            f"{output_suffix}{output_file.suffix}"
        )
    )


def make_cumulative_slope_output_file(output_file, output_suffix=""):
    output_file = Path(output_file)

    return str(
        output_file.with_name(
            f"{output_file.stem}_cumulative_raw_error_slope"
            f"{output_suffix}{output_file.suffix}"
        )
    )


# ============================================================
# AXIS LIMIT HELPERS
# ============================================================

def collect_axis_y_data(ax, logy=False):
    y_values = []

    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]

        if logy:
            y = y[y > 0.0]

        if len(y) > 0:
            y_values.append(y)

    if not y_values:
        return np.array([])

    return np.concatenate(y_values)


def apply_outlier_aware_y_limits(
    ax,
    logy=False,
    main_percentiles=(0.1, 99.9),
    outlier_orders=3.0,
    padding_fraction=0.08,
):
    """
    Apply y-limits that preserve the main data and only clip extreme jumps.

    The old approach used fixed percentile-based clipping, which could cut
    off legitimate portions of the data. This version first checks whether
    the full min/max values are extreme relative to the main data scale.

    If the endpoint is not several orders of magnitude larger than the
    typical/main range, the full data range is shown.

    If the endpoint is several orders of magnitude larger, only then is
    it clipped.
    """

    y = collect_axis_y_data(ax, logy=logy)

    if len(y) == 0:
        return

    if logy:
        y = y[y > 0.0]

    if len(y) == 0:
        return

    full_min = np.nanmin(y)
    full_max = np.nanmax(y)

    if not np.isfinite(full_min) or not np.isfinite(full_max):
        return

    if full_min == full_max:
        if logy:
            ax.set_ylim(full_min * 0.9, full_max * 1.1)
        else:
            pad = abs(full_min) * 0.1
            if pad == 0.0:
                pad = 1.0
            ax.set_ylim(full_min - pad, full_max + pad)
        return

    lower_p, upper_p = main_percentiles
    main_min, main_max = np.percentile(y, [lower_p, upper_p])

    if not np.isfinite(main_min) or not np.isfinite(main_max):
        return

    outlier_factor = 10.0 ** outlier_orders

    visible_min = full_min
    visible_max = full_max

    if logy:
        tiny = np.finfo(float).tiny

        main_min = max(main_min, tiny)
        main_max = max(main_max, tiny)
        full_min = max(full_min, tiny)
        full_max = max(full_max, tiny)

        # Upper clipping for log plots:
        # Only clip if the full maximum is orders of magnitude larger
        # than the upper edge of the main data.
        if full_max > main_max * outlier_factor:
            visible_max = main_max

        # Lower clipping for log plots:
        # Usually less important, but handles extreme tiny dips.
        if full_min < main_min / outlier_factor:
            visible_min = main_min

        if visible_min <= 0.0 or visible_max <= 0.0 or visible_min >= visible_max:
            return

        log_min = np.log10(visible_min)
        log_max = np.log10(visible_max)
        log_range = log_max - log_min

        if log_range <= 0.0:
            return

        log_pad = padding_fraction * log_range

        ax.set_ylim(
            10.0 ** (log_min - log_pad),
            10.0 ** (log_max + log_pad),
        )

    else:
        main_range = main_max - main_min

        if main_range <= 0.0:
            main_scale = max(abs(main_min), abs(main_max), 1.0)
        else:
            main_scale = max(main_range, abs(main_min), abs(main_max), 1.0)

        # Upper clipping for linear plots:
        # Only clip if the maximum is several orders larger than the
        # main data scale.
        if abs(full_max) > abs(main_max) + outlier_factor * main_scale:
            visible_max = main_max

        # Lower clipping for linear plots:
        # Handles extreme negative drops.
        if abs(full_min) > abs(main_min) + outlier_factor * main_scale:
            visible_min = main_min

        if visible_min >= visible_max:
            visible_min = full_min
            visible_max = full_max

        y_range = visible_max - visible_min

        if y_range <= 0.0:
            pad = abs(visible_min) * 0.1
            if pad == 0.0:
                pad = 1.0
        else:
            pad = padding_fraction * y_range

        ax.set_ylim(visible_min - pad, visible_max + pad)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def create_figure(variable_indices):
    num_plots = len(variable_indices)

    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(10, 3.5 * num_plots),
        sharex=True,
        constrained_layout=True,
    )

    if num_plots == 1:
        axes = [axes]

    return fig, axes


def format_axis(ax, var_index, logy=True, show_grid=True):
    ax.set_ylabel(f"var{var_index + 1}", fontsize=12)

    if logy:
        ax.set_yscale("log")

    ax.set_axisbelow(True)

    if show_grid:
        ax.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.6,
            zorder=0,
        )

    ax.legend(fontsize=10, frameon=True)
    ax.tick_params(axis="both", labelsize=11)


def get_output_path(output_file, save_dir="figures"):
    output_file = Path(output_file)

    if output_file.parent == Path("."):
        output_path = Path(save_dir) / output_file
    else:
        output_path = output_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def save_figure(fig, output_file, save_dir="figures"):
    output_path = get_output_path(output_file, save_dir=save_dir)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def get_loaded_data_and_variable_indices(
    filenames,
    selected_vars=None,
    delimiter=",",
    skiprows=0,
):
    filenames = [Path(filename) for filename in filenames]

    loaded_data = []
    max_num_vars = 0

    for filename in filenames:
        iterations, variables = load_data(
            filename,
            delimiter=delimiter,
            skiprows=skiprows,
        )

        loaded_data.append((filename, iterations, variables))
        max_num_vars = max(max_num_vars, variables.shape[1])

    if selected_vars is None:
        variable_indices = list(range(max_num_vars))
    else:
        variable_indices = [var - 1 for var in selected_vars]

    return loaded_data, variable_indices

def plot_files(
    filenames,
    selected_vars=None,
    delimiter=",",
    skiprows=0,
    logy=True,
    title="Data Comparison",
    save_plot=False,
    output_file="plot.png",
    save_dir="figures",
    show_grid=True,
    plot_moving_average=False,
    moving_average_window=10,
    plot_straight_average=True,
):
    """
    Plot one figure containing one or more files.

    Each variable is plotted in its own subplot.
    Files listed together are overlaid on the same axes.

    If enabled, this plots for each file:
        raw data
        moving average of the raw data
        running straight average of the raw data

    Straight average definition:
        For each individual raw data curve, the straight average at the
        current iteration is the arithmetic average of all values from
        the beginning of that curve through the current iteration.

        For a data array:

            y = [1, 2, 3, 4, 5]

        the straight average is:

            running_average = [1, 1.5, 2, 2.5, 3]

        Mathematically:

            running_average[i] = mean(y[0:i+1])

        or equivalently:

            running_average[i] = sum(y[0:i+1]) / (i + 1)

    Important:
        This does NOT average different files together. Each file keeps
        its own separate raw curve, moving-average curve, and straight
        running-average curve.
    """

    loaded_data, variable_indices = get_loaded_data_and_variable_indices(
        filenames=filenames,
        selected_vars=selected_vars,
        delimiter=delimiter,
        skiprows=skiprows,
    )

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):

        for file_idx, (filename, iterations, variables) in enumerate(loaded_data):

            raw_color = FILE_COLORS[file_idx % len(FILE_COLORS)]
            ma_color = MOVING_AVERAGE_COLORS[file_idx % len(MOVING_AVERAGE_COLORS)]

            x = iterations
            y = variables[:, var_index]

            # --------------------------------------------------------
            # Raw data
            # --------------------------------------------------------
            if plot_moving_average:
                raw_linewidth = 1.0
                raw_alpha = 0.70
                raw_zorder = 2
            else:
                raw_linewidth = 2.0
                raw_alpha = 0.85
                raw_zorder = 3

            ax.plot(
                x,
                y,
                color=raw_color,
                linewidth=raw_linewidth,
                alpha=raw_alpha,
                label=filename.stem,
                zorder=raw_zorder,
            )

            # --------------------------------------------------------
            # Moving average of raw data
            # --------------------------------------------------------
            if plot_moving_average:

                if moving_average_window <= len(y):

                    x_avg, y_avg = compute_moving_average(
                        x,
                        y,
                        moving_average_window,
                    )

                    ax.plot(
                        x_avg,
                        y_avg,
                        color=ma_color,
                        linewidth=3.0,
                        alpha=1.0,
                        label=f"{filename.stem} MA-{moving_average_window}",
                        zorder=5,
                    )

                else:
                    print(
                        f"Warning: moving_average_window={moving_average_window} "
                        f"is larger than data length={len(y)} for {filename}. "
                        f"Skipping moving average."
                    )

            # --------------------------------------------------------
            # Straight running average of this file's raw data
            # --------------------------------------------------------
            if plot_straight_average:

                x_running_avg, y_running_avg = compute_running_average(
                    x,
                    y,
                )

                ax.plot(
                    x_running_avg,
                    y_running_avg,
                    color=raw_color,
                    linewidth=2.4,
                    linestyle="-.",
                    alpha=0.95,
                    label=f"{filename.stem} running avg",
                    zorder=6,
                )

        # ------------------------------------------------------------
        # Axis formatting
        # ------------------------------------------------------------
        format_axis(
            ax=ax,
            var_index=var_index,
            logy=logy,
            show_grid=show_grid,
        )

        if APPLY_OUTLIER_AWARE_Y_LIMITS:
            apply_outlier_aware_y_limits(
                ax=ax,
                logy=logy,
                main_percentiles=MAIN_DATA_PERCENTILES,
                outlier_orders=EXTREME_OUTLIER_ORDERS_OF_MAGNITUDE,
                padding_fraction=Y_LIMIT_PADDING_FRACTION,
            )

    axes[-1].set_xlabel("Iteration", fontsize=12)

    if logy:
        figure_title = f"{title} [log-y]"
    else:
        figure_title = title

    if plot_moving_average:
        figure_title = (
            f"{figure_title} "
            f"[MA window = {moving_average_window}]"
        )

    if plot_straight_average:
        figure_title = f"{figure_title} [running average shown]"

    fig.suptitle(figure_title, fontsize=15, fontweight="bold")

    if save_plot:
        save_figure(fig, output_file, save_dir=save_dir)

    return fig, axes

def plot_slope_files(
    filenames,
    selected_vars=None,
    delimiter=",",
    skiprows=0,
    title="Slope Data Comparison",
    save_plot=False,
    output_file="slope_plot.png",
    save_dir="figures",
    show_grid=True,
    plot_moving_average=False,
    moving_average_window=10,
    slope_type="raw",
    slope_logy=False,
    plot_moving_average_of_slope=True,
    plot_magnitude=False,
    plot_slope_curve=True,
    plot_positive_only=False,
):
    loaded_data, variable_indices = get_loaded_data_and_variable_indices(
        filenames=filenames,
        selected_vars=selected_vars,
        delimiter=delimiter,
        skiprows=skiprows,
    )

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):

        for file_idx, (filename, iterations, variables) in enumerate(loaded_data):

            raw_color = FILE_COLORS[file_idx % len(FILE_COLORS)]
            ma_color = MOVING_AVERAGE_COLORS[file_idx % len(MOVING_AVERAGE_COLORS)]

            x = iterations
            y = variables[:, var_index]

            if slope_type == "raw":

                raw_slope = compute_slope(x, y)

                if plot_magnitude:
                    slope_to_plot = np.abs(raw_slope)
                    slope_label = f"{filename.stem} |raw slope|"
                    slope_ma_label = (
                        f"{filename.stem} |raw slope| MA-{moving_average_window}"
                    )

                elif plot_positive_only:
                    slope_to_plot = np.where(raw_slope > 0.0, raw_slope, np.nan)
                    slope_label = f"{filename.stem} positive raw slope"
                    slope_ma_label = (
                        f"{filename.stem} positive raw slope MA-{moving_average_window}"
                    )

                else:
                    slope_to_plot = raw_slope
                    slope_label = f"{filename.stem} raw slope"
                    slope_ma_label = (
                        f"{filename.stem} raw slope MA-{moving_average_window}"
                    )

                if plot_slope_curve:
                    ax.plot(
                        x,
                        slope_to_plot,
                        color=raw_color,
                        linewidth=1.2,
                        alpha=0.55,
                        label=slope_label,
                        zorder=3,
                    )

                if plot_moving_average_of_slope:
                    if moving_average_window <= len(slope_to_plot):

                        if plot_positive_only:
                            x_slope_avg, slope_avg = compute_moving_average_ignore_nan(
                                x,
                                slope_to_plot,
                                moving_average_window,
                            )
                        else:
                            x_slope_avg, slope_avg = compute_moving_average(
                                x,
                                slope_to_plot,
                                moving_average_window,
                            )

                        ax.plot(
                            x_slope_avg,
                            slope_avg,
                            color=ma_color,
                            linewidth=3.0,
                            alpha=1.0,
                            label=slope_ma_label,
                            zorder=6,
                        )
                    else:
                        print(
                            f"Warning: moving_average_window={moving_average_window} "
                            f"is larger than raw slope length={len(slope_to_plot)} "
                            f"for {filename}. Skipping moving average of raw slope."
                        )

            elif slope_type == "ma":

                if plot_moving_average:
                    if moving_average_window <= len(y):

                        x_avg, y_avg = compute_moving_average(
                            x,
                            y,
                            moving_average_window,
                        )

                        ma_slope = compute_slope(x_avg, y_avg)

                        if plot_magnitude:
                            slope_to_plot = np.abs(ma_slope)
                            slope_label = (
                                f"{filename.stem} |MA-{moving_average_window} slope|"
                            )
                            slope_ma_label = (
                                f"{filename.stem} |MA-{moving_average_window} "
                                f"slope| MA-{moving_average_window}"
                            )
                        else:
                            slope_to_plot = ma_slope
                            slope_label = (
                                f"{filename.stem} MA-{moving_average_window} slope"
                            )
                            slope_ma_label = (
                                f"{filename.stem} MA-{moving_average_window} "
                                f"slope MA-{moving_average_window}"
                            )

                        if plot_slope_curve:
                            ax.plot(
                                x_avg,
                                slope_to_plot,
                                color=raw_color,
                                linewidth=1.2,
                                alpha=0.55,
                                label=slope_label,
                                zorder=3,
                            )

                        if plot_moving_average_of_slope:
                            if moving_average_window <= len(slope_to_plot):

                                x_ma_slope_avg, slope_avg = compute_moving_average(
                                    x_avg,
                                    slope_to_plot,
                                    moving_average_window,
                                )

                                ax.plot(
                                    x_ma_slope_avg,
                                    slope_avg,
                                    color=ma_color,
                                    linewidth=3.0,
                                    alpha=1.0,
                                    label=slope_ma_label,
                                    zorder=6,
                                )
                            else:
                                print(
                                    f"Warning: moving_average_window={moving_average_window} "
                                    f"is larger than moving-average slope length="
                                    f"{len(slope_to_plot)} for {filename}. "
                                    f"Skipping moving average of moving-average slope."
                                )
                    else:
                        print(
                            f"Warning: moving_average_window={moving_average_window} "
                            f"is larger than data length={len(y)} for {filename}. "
                            f"Skipping slope of moving average."
                        )

            else:
                raise ValueError("slope_type must be either 'raw' or 'ma'.")

        if not plot_magnitude and not plot_positive_only and plot_slope_curve:
            ax.axhline(
                0.0,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                zorder=1,
            )

        format_axis(
            ax=ax,
            var_index=var_index,
            logy=slope_logy,
            show_grid=show_grid,
        )

        if plot_magnitude:
            ax.set_ylabel(f"|d(var{var_index + 1})/dIteration|", fontsize=12)
        elif plot_positive_only:
            ax.set_ylabel(
                f"positive d(var{var_index + 1})/dIteration",
                fontsize=12,
            )
        else:
            ax.set_ylabel(f"d(var{var_index + 1})/dIteration", fontsize=12)

        if APPLY_OUTLIER_AWARE_Y_LIMITS:
            apply_outlier_aware_y_limits(
                ax=ax,
                logy=slope_logy,
                main_percentiles=MAIN_DATA_PERCENTILES,
                outlier_orders=EXTREME_OUTLIER_ORDERS_OF_MAGNITUDE,
                padding_fraction=Y_LIMIT_PADDING_FRACTION,
            )

    axes[-1].set_xlabel("Iteration", fontsize=12)

    if slope_type == "raw":
        if plot_magnitude:
            figure_title = f"{title} Slope Magnitude [|raw slope|]"
        elif plot_positive_only:
            figure_title = f"{title} Positive Raw Slopes Only"
        elif not plot_slope_curve:
            figure_title = (
                f"{title} Raw Slope Moving Average Only "
                f"[MA window = {moving_average_window}]"
            )
        else:
            figure_title = f"{title} Slope [raw slope]"

    elif slope_type == "ma":
        if plot_magnitude:
            figure_title = (
                f"{title} Slope Magnitude "
                f"[|MA-{moving_average_window} slope|]"
            )
        else:
            figure_title = f"{title} Slope [MA-{moving_average_window} slope]"

    if plot_moving_average_of_slope and plot_slope_curve:
        figure_title = (
            f"{figure_title} "
            f"[slope MA window = {moving_average_window}]"
        )

    fig.suptitle(figure_title, fontsize=15, fontweight="bold")

    if save_plot:
        save_figure(fig, output_file, save_dir=save_dir)

    return fig, axes


def plot_cumulative_slope_files(
    filenames,
    selected_vars=None,
    delimiter=",",
    skiprows=0,
    title="Cumulative Error Slope Comparison",
    save_plot=False,
    output_file="cumulative_slope_plot.png",
    save_dir="figures",
    show_grid=True,
    cumulative_slope_logy=False,
    plot_moving_average_of_cumulative_sum=True,
    moving_average_window=10,
):
    """
    Plot cumulative sum of the raw error slope.

    For each variable:

        raw_slope = d(error)/dIteration
        cumulative_raw_slope[i] = sum(raw_slope[0:i+1])

    If enabled, also plots the moving average of that cumulative sum.
    """

    loaded_data, variable_indices = get_loaded_data_and_variable_indices(
        filenames=filenames,
        selected_vars=selected_vars,
        delimiter=delimiter,
        skiprows=skiprows,
    )

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):

        for file_idx, (filename, iterations, variables) in enumerate(loaded_data):

            raw_color = FILE_COLORS[file_idx % len(FILE_COLORS)]
            ma_color = MOVING_AVERAGE_COLORS[file_idx % len(MOVING_AVERAGE_COLORS)]

            x = iterations
            y = variables[:, var_index]

            raw_slope = compute_slope(x, y)
            cumulative_raw_slope = compute_cumulative_sum(raw_slope)

            ax.plot(
                x,
                cumulative_raw_slope,
                color=raw_color,
                linewidth=1.2,
                alpha=0.55,
                label=f"{filename.stem} cumulative raw slope",
                zorder=3,
            )

            if plot_moving_average_of_cumulative_sum:

                if moving_average_window <= len(cumulative_raw_slope):

                    x_cumulative_avg, cumulative_avg = compute_moving_average(
                        x,
                        cumulative_raw_slope,
                        moving_average_window,
                    )

                    ax.plot(
                        x_cumulative_avg,
                        cumulative_avg,
                        color=ma_color,
                        linewidth=3.0,
                        alpha=1.0,
                        label=(
                            f"{filename.stem} cumulative raw slope "
                            f"MA-{moving_average_window}"
                        ),
                        zorder=6,
                    )

                else:
                    print(
                        f"Warning: moving_average_window={moving_average_window} "
                        f"is larger than cumulative slope length="
                        f"{len(cumulative_raw_slope)} for {filename}. "
                        f"Skipping moving average of cumulative slope."
                    )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        format_axis(
            ax=ax,
            var_index=var_index,
            logy=cumulative_slope_logy,
            show_grid=show_grid,
        )

        ax.set_ylabel(
            f"cumsum d(var{var_index + 1})/dIteration",
            fontsize=12,
        )

        if APPLY_OUTLIER_AWARE_Y_LIMITS:
            apply_outlier_aware_y_limits(
                ax=ax,
                logy=cumulative_slope_logy,
                main_percentiles=MAIN_DATA_PERCENTILES,
                outlier_orders=EXTREME_OUTLIER_ORDERS_OF_MAGNITUDE,
                padding_fraction=Y_LIMIT_PADDING_FRACTION,
            )

    axes[-1].set_xlabel("Iteration", fontsize=12)

    figure_title = f"{title} Cumulative Sum of Raw Error Slope"

    if plot_moving_average_of_cumulative_sum:
        figure_title = (
            f"{figure_title} "
            f"[cumsum MA window = {moving_average_window}]"
        )

    fig.suptitle(figure_title, fontsize=15, fontweight="bold")

    if save_plot:
        save_figure(fig, output_file, save_dir=save_dir)

    return fig, axes


# ============================================================
# MAIN GROUP PLOTTING DRIVER
# ============================================================

def plot_figure_groups(
    figure_groups,
    delimiter=",",
    skiprows=0,
    save_dir="figures",
    pdf_output_file="all_comparisons.pdf",
    show_grid=True,
    show_figures=True,
    plot_moving_average=False,
    moving_average_window=10,
    output_suffix="",
    calculate_slope=False,
    plot_slope_of_raw=True,
    plot_slope_of_moving_average=True,
    plot_slope_magnitude=True,
    slope_logy=False,
    slope_magnitude_logy=False,
    cumulative_slope_logy=False,
    plot_moving_average_of_slope=True,
    plot_raw_slope_moving_average_only=True,
    plot_positive_raw_slopes_only=True,
    plot_cumulative_sum_of_error_slope=True,
    plot_moving_average_of_cumulative_sum=True,
):
    pdf_path = get_output_path(pdf_output_file, save_dir=save_dir)

    with PdfPages(pdf_path) as pdf:

        for group in figure_groups:

            group_output_file = group.get("output_file", "plot.png")

            if output_suffix:
                group_output_file_with_suffix = add_filename_suffix(
                    group_output_file,
                    output_suffix,
                )
            else:
                group_output_file_with_suffix = group_output_file

            # --------------------------------------------------------
            # Main figure
            # --------------------------------------------------------
            fig, axes = plot_files(
                filenames=group["filenames"],
                selected_vars=group.get("selected_vars", None),
                delimiter=delimiter,
                skiprows=group.get("skiprows", skiprows),
                logy=group.get("logy", True),
                title=group.get("title", "Data Comparison"),
                save_plot=group.get("save_plot", False),
                output_file=group_output_file_with_suffix,
                save_dir=save_dir,
                show_grid=show_grid,
                plot_moving_average=group.get(
                    "plot_moving_average",
                    plot_moving_average,
                ),
                moving_average_window=group.get(
                    "moving_average_window",
                    moving_average_window,
                ),
            )

            pdf.savefig(fig, bbox_inches="tight")

            if not show_figures:
                plt.close(fig)

            # --------------------------------------------------------
            # Raw slope figure and related raw-slope figures
            # --------------------------------------------------------
            if (
                group.get("calculate_slope", calculate_slope)
                and group.get("plot_slope_of_raw", plot_slope_of_raw)
            ):

                raw_slope_output_file = make_slope_output_file(
                    group.get("output_file", "plot.png"),
                    slope_type="raw",
                    output_suffix=output_suffix,
                    plot_magnitude=False,
                )

                raw_slope_fig, raw_slope_axes = plot_slope_files(
                    filenames=group["filenames"],
                    selected_vars=group.get("selected_vars", None),
                    delimiter=delimiter,
                    skiprows=group.get("skiprows", skiprows),
                    title=group.get("title", "Data Comparison"),
                    save_plot=group.get("save_plot", False),
                    output_file=raw_slope_output_file,
                    save_dir=save_dir,
                    show_grid=show_grid,
                    plot_moving_average=group.get(
                        "plot_moving_average",
                        plot_moving_average,
                    ),
                    moving_average_window=group.get(
                        "moving_average_window",
                        moving_average_window,
                    ),
                    slope_type="raw",
                    slope_logy=group.get("slope_logy", slope_logy),
                    plot_moving_average_of_slope=group.get(
                        "plot_moving_average_of_slope",
                        plot_moving_average_of_slope,
                    ),
                    plot_magnitude=False,
                    plot_slope_curve=True,
                    plot_positive_only=False,
                )

                pdf.savefig(raw_slope_fig, bbox_inches="tight")

                if not show_figures:
                    plt.close(raw_slope_fig)

                # ----------------------------------------------------
                # Cumulative sum of raw error slope figure
                # ----------------------------------------------------
                if group.get(
                    "plot_cumulative_sum_of_error_slope",
                    plot_cumulative_sum_of_error_slope,
                ):

                    cumulative_slope_output_file = make_cumulative_slope_output_file(
                        group.get("output_file", "plot.png"),
                        output_suffix=output_suffix,
                    )

                    cumulative_slope_fig, cumulative_slope_axes = (
                        plot_cumulative_slope_files(
                            filenames=group["filenames"],
                            selected_vars=group.get("selected_vars", None),
                            delimiter=delimiter,
                            skiprows=group.get("skiprows", skiprows),
                            title=group.get("title", "Data Comparison"),
                            save_plot=group.get("save_plot", False),
                            output_file=cumulative_slope_output_file,
                            save_dir=save_dir,
                            show_grid=show_grid,
                            cumulative_slope_logy=group.get(
                                "cumulative_slope_logy",
                                cumulative_slope_logy,
                            ),
                            plot_moving_average_of_cumulative_sum=group.get(
                                "plot_moving_average_of_cumulative_sum",
                                plot_moving_average_of_cumulative_sum,
                            ),
                            moving_average_window=group.get(
                                "moving_average_window",
                                moving_average_window,
                            ),
                        )
                    )

                    pdf.savefig(cumulative_slope_fig, bbox_inches="tight")

                    if not show_figures:
                        plt.close(cumulative_slope_fig)

                # ----------------------------------------------------
                # Raw slope moving-average-only figure
                # ----------------------------------------------------
                if (
                    group.get(
                        "plot_raw_slope_moving_average_only",
                        plot_raw_slope_moving_average_only,
                    )
                    and group.get(
                        "plot_moving_average_of_slope",
                        plot_moving_average_of_slope,
                    )
                ):

                    raw_slope_ma_only_output_file = (
                        make_raw_slope_moving_average_output_file(
                            group.get("output_file", "plot.png"),
                            output_suffix=output_suffix,
                        )
                    )

                    raw_slope_ma_only_fig, raw_slope_ma_only_axes = plot_slope_files(
                        filenames=group["filenames"],
                        selected_vars=group.get("selected_vars", None),
                        delimiter=delimiter,
                        skiprows=group.get("skiprows", skiprows),
                        title=group.get("title", "Data Comparison"),
                        save_plot=group.get("save_plot", False),
                        output_file=raw_slope_ma_only_output_file,
                        save_dir=save_dir,
                        show_grid=show_grid,
                        plot_moving_average=group.get(
                            "plot_moving_average",
                            plot_moving_average,
                        ),
                        moving_average_window=group.get(
                            "moving_average_window",
                            moving_average_window,
                        ),
                        slope_type="raw",
                        slope_logy=group.get("slope_logy", slope_logy),
                        plot_moving_average_of_slope=True,
                        plot_magnitude=False,
                        plot_slope_curve=False,
                        plot_positive_only=False,
                    )

                    pdf.savefig(raw_slope_ma_only_fig, bbox_inches="tight")

                    if not show_figures:
                        plt.close(raw_slope_ma_only_fig)

                # ----------------------------------------------------
                # Positive raw slope only figure
                # ----------------------------------------------------
                if group.get(
                    "plot_positive_raw_slopes_only",
                    plot_positive_raw_slopes_only,
                ):

                    positive_raw_slope_output_file = (
                        make_positive_raw_slope_output_file(
                            group.get("output_file", "plot.png"),
                            output_suffix=output_suffix,
                        )
                    )

                    positive_raw_slope_fig, positive_raw_slope_axes = (
                        plot_slope_files(
                            filenames=group["filenames"],
                            selected_vars=group.get("selected_vars", None),
                            delimiter=delimiter,
                            skiprows=group.get("skiprows", skiprows),
                            title=group.get("title", "Data Comparison"),
                            save_plot=group.get("save_plot", False),
                            output_file=positive_raw_slope_output_file,
                            save_dir=save_dir,
                            show_grid=show_grid,
                            plot_moving_average=group.get(
                                "plot_moving_average",
                                plot_moving_average,
                            ),
                            moving_average_window=group.get(
                                "moving_average_window",
                                moving_average_window,
                            ),
                            slope_type="raw",
                            slope_logy=group.get("slope_logy", slope_logy),
                            plot_moving_average_of_slope=group.get(
                                "plot_moving_average_of_slope",
                                plot_moving_average_of_slope,
                            ),
                            plot_magnitude=False,
                            plot_slope_curve=True,
                            plot_positive_only=True,
                        )
                    )

                    pdf.savefig(positive_raw_slope_fig, bbox_inches="tight")

                    if not show_figures:
                        plt.close(positive_raw_slope_fig)

                # ----------------------------------------------------
                # Raw slope magnitude figure
                # ----------------------------------------------------
                if group.get("plot_slope_magnitude", plot_slope_magnitude):

                    raw_slope_magnitude_output_file = make_slope_output_file(
                        group.get("output_file", "plot.png"),
                        slope_type="raw",
                        output_suffix=output_suffix,
                        plot_magnitude=True,
                    )

                    raw_slope_magnitude_fig, raw_slope_magnitude_axes = (
                        plot_slope_files(
                            filenames=group["filenames"],
                            selected_vars=group.get("selected_vars", None),
                            delimiter=delimiter,
                            skiprows=group.get("skiprows", skiprows),
                            title=group.get("title", "Data Comparison"),
                            save_plot=group.get("save_plot", False),
                            output_file=raw_slope_magnitude_output_file,
                            save_dir=save_dir,
                            show_grid=show_grid,
                            plot_moving_average=group.get(
                                "plot_moving_average",
                                plot_moving_average,
                            ),
                            moving_average_window=group.get(
                                "moving_average_window",
                                moving_average_window,
                            ),
                            slope_type="raw",
                            slope_logy=group.get(
                                "slope_magnitude_logy",
                                slope_magnitude_logy,
                            ),
                            plot_moving_average_of_slope=group.get(
                                "plot_moving_average_of_slope",
                                plot_moving_average_of_slope,
                            ),
                            plot_magnitude=True,
                            plot_slope_curve=True,
                            plot_positive_only=False,
                        )
                    )

                    pdf.savefig(raw_slope_magnitude_fig, bbox_inches="tight")

                    if not show_figures:
                        plt.close(raw_slope_magnitude_fig)

            # --------------------------------------------------------
            # Moving-average slope figures
            # --------------------------------------------------------
            if (
                group.get("calculate_slope", calculate_slope)
                and group.get(
                    "plot_slope_of_moving_average",
                    plot_slope_of_moving_average,
                )
            ):

                ma_slope_output_file = make_slope_output_file(
                    group.get("output_file", "plot.png"),
                    slope_type="ma",
                    output_suffix=output_suffix,
                    plot_magnitude=False,
                )

                ma_slope_fig, ma_slope_axes = plot_slope_files(
                    filenames=group["filenames"],
                    selected_vars=group.get("selected_vars", None),
                    delimiter=delimiter,
                    skiprows=group.get("skiprows", skiprows),
                    title=group.get("title", "Data Comparison"),
                    save_plot=group.get("save_plot", False),
                    output_file=ma_slope_output_file,
                    save_dir=save_dir,
                    show_grid=show_grid,
                    plot_moving_average=group.get(
                        "plot_moving_average",
                        plot_moving_average,
                    ),
                    moving_average_window=group.get(
                        "moving_average_window",
                        moving_average_window,
                    ),
                    slope_type="ma",
                    slope_logy=group.get("slope_logy", slope_logy),
                    plot_moving_average_of_slope=group.get(
                        "plot_moving_average_of_slope",
                        plot_moving_average_of_slope,
                    ),
                    plot_magnitude=False,
                    plot_slope_curve=True,
                    plot_positive_only=False,
                )

                pdf.savefig(ma_slope_fig, bbox_inches="tight")

                if not show_figures:
                    plt.close(ma_slope_fig)

                # ----------------------------------------------------
                # Moving-average slope magnitude figure
                # ----------------------------------------------------
                if group.get("plot_slope_magnitude", plot_slope_magnitude):

                    ma_slope_magnitude_output_file = make_slope_output_file(
                        group.get("output_file", "plot.png"),
                        slope_type="ma",
                        output_suffix=output_suffix,
                        plot_magnitude=True,
                    )

                    ma_slope_magnitude_fig, ma_slope_magnitude_axes = (
                        plot_slope_files(
                            filenames=group["filenames"],
                            selected_vars=group.get("selected_vars", None),
                            delimiter=delimiter,
                            skiprows=group.get("skiprows", skiprows),
                            title=group.get("title", "Data Comparison"),
                            save_plot=group.get("save_plot", False),
                            output_file=ma_slope_magnitude_output_file,
                            save_dir=save_dir,
                            show_grid=show_grid,
                            plot_moving_average=group.get(
                                "plot_moving_average",
                                plot_moving_average,
                            ),
                            moving_average_window=group.get(
                                "moving_average_window",
                                moving_average_window,
                            ),
                            slope_type="ma",
                            slope_logy=group.get(
                                "slope_magnitude_logy",
                                slope_magnitude_logy,
                            ),
                            plot_moving_average_of_slope=group.get(
                                "plot_moving_average_of_slope",
                                plot_moving_average_of_slope,
                            ),
                            plot_magnitude=True,
                            plot_slope_curve=True,
                            plot_positive_only=False,
                        )
                    )

                    pdf.savefig(ma_slope_magnitude_fig, bbox_inches="tight")

                    if not show_figures:
                        plt.close(ma_slope_magnitude_fig)

    print(f"Saved combined PDF to: {pdf_path}")

    if show_figures:
        plt.show()
    else:
        plt.close("all")


# ============================================================
# MAIN
# ============================================================

def main():

    if PLOT_MOVING_AVERAGE:

        for window in MOVING_AVERAGE_WINDOWS:

            print("")
            print("============================================================")
            print(f"Creating plots for moving-average window = {window}")
            print("============================================================")

            window_save_dir = Path(SAVE_DIR) / f"MA_{window}"

            output_suffix = f"_MA_{window}"

            pdf_output_file = add_filename_suffix(
                PDF_OUTPUT_FILE,
                output_suffix,
            )

            plot_figure_groups(
                figure_groups=FIGURE_GROUPS,
                delimiter=DELIMITER,
                skiprows=SKIPROWS,
                save_dir=window_save_dir,
                pdf_output_file=pdf_output_file,
                show_grid=SHOW_GRID,
                show_figures=SHOW_FIGURES,
                plot_moving_average=True,
                moving_average_window=window,
                output_suffix=output_suffix,
                calculate_slope=CALCULATE_SLOPE,
                plot_slope_of_raw=PLOT_SLOPE_OF_RAW,
                plot_slope_of_moving_average=PLOT_SLOPE_OF_MOVING_AVERAGE,
                plot_slope_magnitude=PLOT_SLOPE_MAGNITUDE,
                slope_logy=SLOPE_LOGY,
                slope_magnitude_logy=SLOPE_MAGNITUDE_LOGY,
                cumulative_slope_logy=CUMULATIVE_SLOPE_LOGY,
                plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                plot_raw_slope_moving_average_only=(
                    PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY
                ),
                plot_positive_raw_slopes_only=PLOT_POSITIVE_RAW_SLOPES_ONLY,
                plot_cumulative_sum_of_error_slope=(
                    PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE
                ),
                plot_moving_average_of_cumulative_sum=(
                    PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM
                ),
            )

    else:

        plot_figure_groups(
            figure_groups=FIGURE_GROUPS,
            delimiter=DELIMITER,
            skiprows=SKIPROWS,
            save_dir=SAVE_DIR,
            pdf_output_file=PDF_OUTPUT_FILE,
            show_grid=SHOW_GRID,
            show_figures=SHOW_FIGURES,
            plot_moving_average=False,
            moving_average_window=10,
            output_suffix="",
            calculate_slope=CALCULATE_SLOPE,
            plot_slope_of_raw=PLOT_SLOPE_OF_RAW,
            plot_slope_of_moving_average=False,
            plot_slope_magnitude=PLOT_SLOPE_MAGNITUDE,
            slope_logy=SLOPE_LOGY,
            slope_magnitude_logy=SLOPE_MAGNITUDE_LOGY,
            cumulative_slope_logy=CUMULATIVE_SLOPE_LOGY,
            plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
            plot_raw_slope_moving_average_only=(
                PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY
            ),
            plot_positive_raw_slopes_only=PLOT_POSITIVE_RAW_SLOPES_ONLY,
            plot_cumulative_sum_of_error_slope=(
                PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE
            ),
            plot_moving_average_of_cumulative_sum=(
                PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM
            ),
        )


if __name__ == "__main__":
    main()