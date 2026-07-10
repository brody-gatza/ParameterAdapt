#!/usr/bin/env python3

from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# USER SETTINGS
# ============================================================

DELIMITER = ","
SKIPROWS = 50

OUTPUT_DIR = Path("figures")
PDF_OUTPUT_FILE = "all_comparisons.pdf"

CLEAN_OUTPUT_DIR_BEFORE_WRITING = True

SHOW_GRID = True
SHOW_FIGURES = False

PLOT_MOVING_AVERAGE = True
MOVING_AVERAGE_WINDOWS = [1000]

PLOT_RUNNING_AVERAGE = True

CALCULATE_SLOPE = True

PLOT_SLOPE_OF_RAW = True
PLOT_SLOPE_OF_MOVING_AVERAGE = True
PLOT_SLOPE_MAGNITUDE = True
PLOT_MOVING_AVERAGE_OF_SLOPE = True
PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY = True
PLOT_POSITIVE_RAW_SLOPES_ONLY = True

PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE = True
PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM = True

SLOPE_LOGY = False
SLOPE_MAGNITUDE_LOGY = False
CUMULATIVE_SLOPE_LOGY = False


# ============================================================
# Y-AXIS LIMIT SETTINGS
# ============================================================
#
# These settings only affect the visible y-axis range.
# The plotted data itself is not modified.
#
# The y-limits are determined from the data itself using an adaptive
# outlier fence. Large spikes beyond the non-outlier data range can go
# outside the plot limits.
#
# For log-y plots, outlier detection is done in log10 space.

APPLY_AUTO_Y_LIMITS = True

# Padding added around the final visible y-range.
Y_LIMIT_PADDING_FRACTION = 0.08


# ============================================================
# COLOR SETTINGS
# ============================================================
#
# Typical usage:
#   PRIMARY_COLORS   -> raw/main curve
#   SECONDARY_COLORS -> moving average / smoothed curve
#   TERTIARY_COLORS  -> running average / alternate derived curve

PRIMARY_COLORS = [
    "blue",
    "orange",
    "green",
    "purple",
]

SECONDARY_COLORS = [
    "cyan",
    "red",
    "lime",
    "magenta",
]

TERTIARY_COLORS = [
    "navy",
    "darkorange",
    "darkgreen",
    "indigo",
]


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
    },
    {
        "title": "Primitive Variables - Avg",
        "filenames": [
            "prim_interp_avg.txt",
            "prim_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Primitive Variables - Min",
        "filenames": [
            "prim_interp_min.txt",
            "prim_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
    },
    {
        "title": "Conservative Variables - Max",
        "filenames": [
            "cons_interp_max.txt",
            "cons_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Conservative Variables - Avg",
        "filenames": [
            "cons_interp_avg.txt",
            "cons_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Conservative Variables - Min",
        "filenames": [
            "cons_interp_min.txt",
            "cons_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
    },
    {
        "title": "Gradient Primitive Variables - Max",
        "filenames": [
            "grad_prim_interp_max.txt",
            "grad_prim_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Gradient Primitive Variables - Avg",
        "filenames": [
            "grad_prim_interp_avg.txt",
            "grad_prim_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Gradient Primitive Variables - Min",
        "filenames": [
            "grad_prim_interp_min.txt",
            "grad_prim_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
    },
    {
        "title": "Gradient Conservative Variables - Max",
        "filenames": [
            "grad_cons_interp_max.txt",
            "grad_cons_proj_max.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Gradient Conservative Variables - Avg",
        "filenames": [
            "grad_cons_interp_avg.txt",
            "grad_cons_proj_avg.txt",
        ],
        "selected_vars": None,
        "logy": True,
    },
    {
        "title": "Gradient Conservative Variables - Min",
        "filenames": [
            "grad_cons_interp_min.txt",
            "grad_cons_proj_min.txt",
        ],
        "selected_vars": None,
        "logy": False,
    },
]


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(filename, delimiter=DELIMITER, skiprows=SKIPROWS):
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

    iterations = data[:, 0]
    variables = data[:, 1:]

    return iterations, variables


def load_group_data(group, delimiter=DELIMITER, skiprows=SKIPROWS):
    loaded_data = []
    max_num_vars = 0

    for filename in group["filenames"]:
        filename = Path(filename)

        iterations, variables = load_data(
            filename,
            delimiter=delimiter,
            skiprows=group.get("skiprows", skiprows),
        )

        loaded_data.append(
            {
                "filename": filename,
                "iterations": iterations,
                "variables": variables,
            }
        )

        max_num_vars = max(max_num_vars, variables.shape[1])

    selected_vars = group.get("selected_vars")

    if selected_vars is None:
        variable_indices = list(range(max_num_vars))
    else:
        variable_indices = [index - 1 for index in selected_vars]

    return loaded_data, variable_indices


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
    kernel = np.ones(window)

    valid_count = np.convolve(valid.astype(float), kernel, mode="valid")
    y_sum = np.convolve(y_clean, kernel, mode="valid")

    with np.errstate(divide="ignore", invalid="ignore"):
        y_avg = y_sum / valid_count

    y_avg[valid_count == 0] = np.nan

    start = window // 2
    end = start + len(y_avg)

    x_avg = x[start:end]

    return x_avg, y_avg


def compute_running_average(x, y):
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
# Y-AXIS LIMIT HELPERS
# ============================================================

def collect_line_y_data(line, logy=False):
    """
    Collect valid y-data from one plotted line.
    """

    y = np.asarray(line.get_ydata(), dtype=float)
    y = y[np.isfinite(y)]

    if logy:
        y = y[y > 0.0]

    return y


def get_adaptive_outlier_fence_limits(y, logy=False):
    """
    Compute visible y-limits for one plotted line using an adaptive
    data-based outlier fence.

    The plotted data is not modified.

    For log-y plots:
        - Work in log10(y)
        - Determine the non-outlier range in log space
        - Convert the visible limits back to normal y-values

    For linear plots:
        - Work directly with y

    This avoids user-selected clipping thresholds like 10x, 40x, or 100x.
    The cutoff adapts to the distribution of each plotted curve.
    """

    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if logy:
        y = y[y > 0.0]

    if len(y) == 0:
        return None

    if len(y) < 4:
        y_min = np.min(y)
        y_max = np.max(y)

        if y_min == y_max:
            if logy:
                return y_min * 0.9, y_max * 1.1

            pad = abs(y_min) * 0.1

            if pad == 0.0:
                pad = 1.0

            return y_min - pad, y_max + pad

        return y_min, y_max

    if logy:
        working_y = np.log10(y)
    else:
        working_y = y.copy()

    working_y = working_y[np.isfinite(working_y)]

    if len(working_y) == 0:
        return None

    q1 = np.percentile(working_y, 25.0)
    q3 = np.percentile(working_y, 75.0)
    iqr = q3 - q1

    if not np.isfinite(q1) or not np.isfinite(q3) or not np.isfinite(iqr):
        return None

    # ------------------------------------------------------------
    # Adaptive outlier fence
    # ------------------------------------------------------------
    #
    # This is intentionally an outer fence, not the more aggressive
    # standard boxplot 1.5*IQR fence.
    #
    # It is meant to preserve medium spikes while removing isolated
    # extreme spikes from the axis-limit calculation.
    #
    # This multiplier is internal on purpose; the visible cutoff is still
    # determined from the dataset spread, not from a user-selected jump ratio.
    # ------------------------------------------------------------
    outer_fence_multiplier = 3.0

    if iqr == 0.0:
        center = np.median(working_y)
        deviations = np.abs(working_y - center)
        nonzero_deviations = deviations[deviations > 0.0]

        if len(nonzero_deviations) == 0:
            if logy:
                visible_min = 10.0 ** (center - 0.5)
                visible_max = 10.0 ** (center + 0.5)

                return visible_min, visible_max

            pad = abs(center) * 0.1

            if pad == 0.0:
                pad = 1.0

            return center - pad, center + pad

        spread = np.median(nonzero_deviations)

        lower_fence = center - outer_fence_multiplier * spread
        upper_fence = center + outer_fence_multiplier * spread

    else:
        lower_fence = q1 - outer_fence_multiplier * iqr
        upper_fence = q3 + outer_fence_multiplier * iqr

    visible_working_y = working_y[
        (working_y >= lower_fence)
        & (working_y <= upper_fence)
    ]

    # If the fence removed everything, fall back to the full data.
    if len(visible_working_y) == 0:
        visible_working_y = working_y

    visible_min_working = np.min(visible_working_y)
    visible_max_working = np.max(visible_working_y)

    if not np.isfinite(visible_min_working) or not np.isfinite(visible_max_working):
        return None

    if visible_min_working == visible_max_working:
        if logy:
            visible_min_working -= 0.5
            visible_max_working += 0.5
        else:
            pad = abs(visible_min_working) * 0.1

            if pad == 0.0:
                pad = 1.0

            visible_min_working -= pad
            visible_max_working += pad

    if visible_min_working >= visible_max_working:
        return None

    if logy:
        visible_min = 10.0 ** visible_min_working
        visible_max = 10.0 ** visible_max_working

        if visible_min <= 0.0 or visible_max <= 0.0:
            return None

        if visible_min >= visible_max:
            return None

        return visible_min, visible_max

    return visible_min_working, visible_max_working


def apply_adaptive_y_limits(
    ax,
    logy=False,
    padding_fraction=Y_LIMIT_PADDING_FRACTION,
    keep_zero_visible=False,
):
    """
    Set y-limits using adaptive per-line outlier fences.

    Each plotted line gets its own non-outlier visible range.
    The subplot then uses the union of those per-line ranges.

    This means:
        - raw data can still control the range if it has the widest
          non-outlier range
        - moving averages cannot overly tighten the axis
        - medium spikes are preserved
        - extreme isolated spikes can go outside the visible y-limits
    """

    line_limits = []

    for line in ax.get_lines():
        y = collect_line_y_data(line, logy=logy)

        if len(y) == 0:
            continue

        limits = get_adaptive_outlier_fence_limits(
            y,
            logy=logy,
        )

        if limits is not None:
            line_limits.append(limits)

    if not line_limits:
        return

    visible_min = min(limits[0] for limits in line_limits)
    visible_max = max(limits[1] for limits in line_limits)

    if not np.isfinite(visible_min) or not np.isfinite(visible_max):
        return

    if visible_min >= visible_max:
        return

    if logy:
        if visible_min <= 0.0 or visible_max <= 0.0:
            return

        log_min = np.log10(visible_min)
        log_max = np.log10(visible_max)
        log_range = log_max - log_min

        if log_range <= 0.0:
            return

        log_pad = padding_fraction * log_range

        final_min = 10.0 ** (log_min - log_pad)
        final_max = 10.0 ** (log_max + log_pad)

    else:
        y_range = visible_max - visible_min

        if y_range <= 0.0:
            return

        pad = padding_fraction * y_range

        final_min = visible_min - pad
        final_max = visible_max + pad

        if keep_zero_visible:
            if final_min > 0.0:
                final_min = 0.0
            elif final_max < 0.0:
                final_max = 0.0

    if final_min >= final_max:
        return

    ax.set_ylim(final_min, final_max)


def apply_limits_if_enabled(ax, logy=False, keep_zero_visible=False):
    if APPLY_AUTO_Y_LIMITS:
        apply_adaptive_y_limits(
            ax=ax,
            logy=logy,
            padding_fraction=Y_LIMIT_PADDING_FRACTION,
            keep_zero_visible=keep_zero_visible,
        )


# ============================================================
# PLOTTING HELPERS
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


def format_axis(ax, ylabel, logy=False, show_grid=SHOW_GRID):
    ax.set_ylabel(ylabel, fontsize=12)

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


def finalize_figure(fig, axes, title):
    axes[-1].set_xlabel("Iteration", fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold")


def add_page_to_pdf(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")

    if not SHOW_FIGURES:
        plt.close(fig)


# ============================================================
# MAIN DATA PLOT
# ============================================================

def plot_main_data(
    group,
    moving_average_window=None,
    plot_moving_average=False,
    plot_running_average=PLOT_RUNNING_AVERAGE,
):
    loaded_data, variable_indices = load_group_data(group)

    logy = group.get("logy", True)
    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for file_idx, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            primary_color = PRIMARY_COLORS[file_idx % len(PRIMARY_COLORS)]
            secondary_color = SECONDARY_COLORS[file_idx % len(SECONDARY_COLORS)]
            tertiary_color = TERTIARY_COLORS[file_idx % len(TERTIARY_COLORS)]

            raw_linewidth = 1.0 if plot_moving_average else 2.0
            raw_alpha = 0.70 if plot_moving_average else 0.85

            ax.plot(
                x,
                y,
                color=primary_color,
                linewidth=raw_linewidth,
                alpha=raw_alpha,
                label=filename.stem,
                zorder=3,
            )

            if plot_moving_average:
                if moving_average_window is None:
                    continue

                if moving_average_window <= len(y):
                    x_avg, y_avg = compute_moving_average(
                        x,
                        y,
                        moving_average_window,
                    )

                    ax.plot(
                        x_avg,
                        y_avg,
                        color=secondary_color,
                        linewidth=3.0,
                        alpha=1.0,
                        label=f"{filename.stem} MA-{moving_average_window}",
                        zorder=5,
                    )
                else:
                    print(
                        f"Warning: MA window={moving_average_window} "
                        f"is larger than data length={len(y)} for {filename}. "
                        "Skipping moving average."
                    )

            if plot_running_average:
                x_running, y_running = compute_running_average(x, y)

                ax.plot(
                    x_running,
                    y_running,
                    color=tertiary_color,
                    linewidth=2.4,
                    linestyle="-.",
                    alpha=0.95,
                    label=f"{filename.stem} running avg",
                    zorder=6,
                )

        format_axis(
            ax,
            ylabel=f"var{var_index + 1}",
            logy=logy,
        )

        apply_limits_if_enabled(
            ax,
            logy=logy,
            keep_zero_visible=False,
        )

    figure_title = title

    if logy:
        figure_title += " [log-y]"

    if plot_moving_average:
        figure_title += f" [MA window = {moving_average_window}]"

    if plot_running_average:
        figure_title += " [running average shown]"

    finalize_figure(fig, axes, figure_title)

    return fig


# ============================================================
# SLOPE PLOTS
# ============================================================

def plot_slope_data(
    group,
    moving_average_window,
    slope_source="raw",
    plot_magnitude=False,
    plot_positive_only=False,
    plot_slope_curve=True,
    plot_moving_average_of_slope=True,
    slope_logy=False,
):
    loaded_data, variable_indices = load_group_data(group)

    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for file_idx, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            primary_color = PRIMARY_COLORS[file_idx % len(PRIMARY_COLORS)]
            secondary_color = SECONDARY_COLORS[file_idx % len(SECONDARY_COLORS)]

            if slope_source == "raw":
                slope_x = x
                slope_y = compute_slope(x, y)
                source_label = "raw slope"

            elif slope_source == "ma":
                if moving_average_window is None:
                    print(
                        f"Warning: moving_average_window is None for {filename}. "
                        "Skipping slope of moving average."
                    )
                    continue

                if moving_average_window > len(y):
                    print(
                        f"Warning: MA window={moving_average_window} "
                        f"is larger than data length={len(y)} for {filename}. "
                        "Skipping slope of moving average."
                    )
                    continue

                x_avg, y_avg = compute_moving_average(
                    x,
                    y,
                    moving_average_window,
                )

                slope_x = x_avg
                slope_y = compute_slope(x_avg, y_avg)
                source_label = f"MA-{moving_average_window} slope"

            else:
                raise ValueError("slope_source must be either 'raw' or 'ma'.")

            if plot_magnitude:
                slope_y = np.abs(slope_y)
                line_label = f"{filename.stem} |{source_label}|"
                ma_label = (
                    f"{filename.stem} |{source_label}| "
                    f"MA-{moving_average_window}"
                )

            elif plot_positive_only:
                slope_y = np.where(slope_y > 0.0, slope_y, np.nan)
                line_label = f"{filename.stem} positive {source_label}"
                ma_label = (
                    f"{filename.stem} positive {source_label} "
                    f"MA-{moving_average_window}"
                )

            else:
                line_label = f"{filename.stem} {source_label}"
                ma_label = (
                    f"{filename.stem} {source_label} "
                    f"MA-{moving_average_window}"
                )

            if plot_slope_curve:
                ax.plot(
                    slope_x,
                    slope_y,
                    color=primary_color,
                    linewidth=1.2,
                    alpha=0.55,
                    label=line_label,
                    zorder=3,
                )

            if plot_moving_average_of_slope:
                if moving_average_window is None:
                    print(
                        f"Warning: moving_average_window is None for {filename}. "
                        "Skipping moving average of slope."
                    )
                    continue

                if moving_average_window <= len(slope_y):
                    if plot_positive_only:
                        x_slope_avg, slope_avg = compute_moving_average_ignore_nan(
                            slope_x,
                            slope_y,
                            moving_average_window,
                        )
                    else:
                        x_slope_avg, slope_avg = compute_moving_average(
                            slope_x,
                            slope_y,
                            moving_average_window,
                        )

                    ax.plot(
                        x_slope_avg,
                        slope_avg,
                        color=secondary_color,
                        linewidth=3.0,
                        alpha=1.0,
                        label=ma_label,
                        zorder=6,
                    )
                else:
                    print(
                        f"Warning: MA window={moving_average_window} "
                        f"is larger than slope length={len(slope_y)} for {filename}. "
                        "Skipping moving average of slope."
                    )

        if not plot_magnitude and not plot_positive_only and plot_slope_curve:
            ax.axhline(
                0.0,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                zorder=1,
            )

        if plot_magnitude:
            ylabel = f"|d(var{var_index + 1})/dIteration|"
        elif plot_positive_only:
            ylabel = f"positive d(var{var_index + 1})/dIteration"
        else:
            ylabel = f"d(var{var_index + 1})/dIteration"

        format_axis(
            ax,
            ylabel=ylabel,
            logy=slope_logy,
        )

        apply_limits_if_enabled(
            ax,
            logy=slope_logy,
            keep_zero_visible=not slope_logy,
        )

    figure_title = title

    if slope_source == "raw":
        if plot_magnitude:
            figure_title += " Slope Magnitude [|raw slope|]"
        elif plot_positive_only:
            figure_title += " Positive Raw Slopes Only"
        elif not plot_slope_curve:
            figure_title += (
                f" Raw Slope Moving Average Only "
                f"[MA window = {moving_average_window}]"
            )
        else:
            figure_title += " Slope [raw slope]"

    elif slope_source == "ma":
        if plot_magnitude:
            figure_title += (
                f" Slope Magnitude "
                f"[|MA-{moving_average_window} slope|]"
            )
        else:
            figure_title += f" Slope [MA-{moving_average_window} slope]"

    if plot_moving_average_of_slope and plot_slope_curve:
        figure_title += f" [slope MA window = {moving_average_window}]"

    finalize_figure(fig, axes, figure_title)

    return fig


# ============================================================
# CUMULATIVE SLOPE PLOTS
# ============================================================

def plot_cumulative_slope_data(
    group,
    moving_average_window,
    cumulative_slope_logy=False,
    plot_moving_average_of_cumulative_sum=True,
):
    loaded_data, variable_indices = load_group_data(group)

    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for file_idx, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            primary_color = PRIMARY_COLORS[file_idx % len(PRIMARY_COLORS)]
            secondary_color = SECONDARY_COLORS[file_idx % len(SECONDARY_COLORS)]

            raw_slope = compute_slope(x, y)
            cumulative_raw_slope = compute_cumulative_sum(raw_slope)

            ax.plot(
                x,
                cumulative_raw_slope,
                color=primary_color,
                linewidth=1.2,
                alpha=0.55,
                label=f"{filename.stem} cumulative raw slope",
                zorder=3,
            )

            if plot_moving_average_of_cumulative_sum:
                if moving_average_window is None:
                    print(
                        f"Warning: moving_average_window is None for {filename}. "
                        "Skipping cumulative slope moving average."
                    )
                    continue

                if moving_average_window <= len(cumulative_raw_slope):
                    x_avg, cumulative_avg = compute_moving_average(
                        x,
                        cumulative_raw_slope,
                        moving_average_window,
                    )

                    ax.plot(
                        x_avg,
                        cumulative_avg,
                        color=secondary_color,
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
                        f"Warning: MA window={moving_average_window} "
                        f"is larger than cumulative slope length="
                        f"{len(cumulative_raw_slope)} for {filename}. "
                        "Skipping cumulative slope MA."
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
            ax,
            ylabel=f"cumsum d(var{var_index + 1})/dIteration",
            logy=cumulative_slope_logy,
        )

        apply_limits_if_enabled(
            ax,
            logy=cumulative_slope_logy,
            keep_zero_visible=not cumulative_slope_logy,
        )

    figure_title = f"{title} Cumulative Sum of Raw Error Slope"

    if plot_moving_average_of_cumulative_sum:
        figure_title += f" [cumsum MA window = {moving_average_window}]"

    finalize_figure(fig, axes, figure_title)

    return fig


# ============================================================
# PDF / OUTPUT HELPERS
# ============================================================

def make_pdf_name(base_name, moving_average_window=None):
    base_name = Path(base_name)

    if moving_average_window is None:
        return base_name

    return base_name.with_name(
        f"{base_name.stem}_MA_{moving_average_window}{base_name.suffix}"
    )


def clean_output_directory(output_dir):
    """
    Delete the output directory before writing new results.

    This prevents old PDFs from previous runs from remaining in the folder.
    Safety checks are included to avoid deleting dangerous paths.
    """

    output_dir = Path(output_dir)

    if not output_dir.exists():
        return

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path exists but is not a directory: {output_dir}"
        )

    resolved_output_dir = output_dir.resolve()
    current_working_dir = Path.cwd().resolve()

    unsafe_paths = {
        Path("/").resolve(),
        current_working_dir,
        current_working_dir.parent,
        Path.home().resolve(),
    }

    if resolved_output_dir in unsafe_paths:
        raise ValueError(
            f"Refusing to delete unsafe output directory: {resolved_output_dir}"
        )

    print(f"Deleting existing output directory: {resolved_output_dir}")
    shutil.rmtree(resolved_output_dir)


def write_pdf(pdf_path, moving_average_window=None, plot_moving_average=False):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    print("")
    print("============================================================")

    if moving_average_window is None:
        print(f"Creating PDF: {pdf_path}")
    else:
        print(f"Creating PDF for moving-average window = {moving_average_window}")
        print(f"Output: {pdf_path}")

    print("============================================================")

    with PdfPages(pdf_path) as pdf:
        for group in FIGURE_GROUPS:
            group_title = group.get("title", "Data Comparison")
            print(f"Adding group: {group_title}")

            fig = plot_main_data(
                group,
                moving_average_window=moving_average_window,
                plot_moving_average=plot_moving_average,
                plot_running_average=PLOT_RUNNING_AVERAGE,
            )
            add_page_to_pdf(pdf, fig)

            if not CALCULATE_SLOPE:
                continue

            if PLOT_SLOPE_OF_RAW:
                fig = plot_slope_data(
                    group,
                    moving_average_window=moving_average_window,
                    slope_source="raw",
                    plot_magnitude=False,
                    plot_positive_only=False,
                    plot_slope_curve=True,
                    plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                    slope_logy=SLOPE_LOGY,
                )
                add_page_to_pdf(pdf, fig)

                if PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE:
                    fig = plot_cumulative_slope_data(
                        group,
                        moving_average_window=moving_average_window,
                        cumulative_slope_logy=CUMULATIVE_SLOPE_LOGY,
                        plot_moving_average_of_cumulative_sum=(
                            PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM
                        ),
                    )
                    add_page_to_pdf(pdf, fig)

                if PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY:
                    if PLOT_MOVING_AVERAGE_OF_SLOPE:
                        fig = plot_slope_data(
                            group,
                            moving_average_window=moving_average_window,
                            slope_source="raw",
                            plot_magnitude=False,
                            plot_positive_only=False,
                            plot_slope_curve=False,
                            plot_moving_average_of_slope=True,
                            slope_logy=SLOPE_LOGY,
                        )
                        add_page_to_pdf(pdf, fig)

                if PLOT_POSITIVE_RAW_SLOPES_ONLY:
                    fig = plot_slope_data(
                        group,
                        moving_average_window=moving_average_window,
                        slope_source="raw",
                        plot_magnitude=False,
                        plot_positive_only=True,
                        plot_slope_curve=True,
                        plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                        slope_logy=SLOPE_LOGY,
                    )
                    add_page_to_pdf(pdf, fig)

                if PLOT_SLOPE_MAGNITUDE:
                    fig = plot_slope_data(
                        group,
                        moving_average_window=moving_average_window,
                        slope_source="raw",
                        plot_magnitude=True,
                        plot_positive_only=False,
                        plot_slope_curve=True,
                        plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                        slope_logy=SLOPE_MAGNITUDE_LOGY,
                    )
                    add_page_to_pdf(pdf, fig)

            if plot_moving_average and PLOT_SLOPE_OF_MOVING_AVERAGE:
                fig = plot_slope_data(
                    group,
                    moving_average_window=moving_average_window,
                    slope_source="ma",
                    plot_magnitude=False,
                    plot_positive_only=False,
                    plot_slope_curve=True,
                    plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                    slope_logy=SLOPE_LOGY,
                )
                add_page_to_pdf(pdf, fig)

                if PLOT_SLOPE_MAGNITUDE:
                    fig = plot_slope_data(
                        group,
                        moving_average_window=moving_average_window,
                        slope_source="ma",
                        plot_magnitude=True,
                        plot_positive_only=False,
                        plot_slope_curve=True,
                        plot_moving_average_of_slope=PLOT_MOVING_AVERAGE_OF_SLOPE,
                        slope_logy=SLOPE_MAGNITUDE_LOGY,
                    )
                    add_page_to_pdf(pdf, fig)

    print(f"Saved PDF to: {pdf_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if CLEAN_OUTPUT_DIR_BEFORE_WRITING:
        clean_output_directory(OUTPUT_DIR)

    if PLOT_MOVING_AVERAGE:
        for window in MOVING_AVERAGE_WINDOWS:
            pdf_name = make_pdf_name(
                PDF_OUTPUT_FILE,
                moving_average_window=window,
            )

            pdf_path = OUTPUT_DIR / pdf_name

            write_pdf(
                pdf_path=pdf_path,
                moving_average_window=window,
                plot_moving_average=True,
            )

    else:
        pdf_path = OUTPUT_DIR / PDF_OUTPUT_FILE

        write_pdf(
            pdf_path=pdf_path,
            moving_average_window=None,
            plot_moving_average=False,
        )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()