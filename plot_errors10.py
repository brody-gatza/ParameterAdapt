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
SKIPROWS = 0

OUTPUT_DIR = Path("figures")
PDF_OUTPUT_FILE = "all_comparisons.pdf"

CLEAN_OUTPUT_DIR_BEFORE_WRITING = True

SHOW_GRID = True
SHOW_FIGURES = False

PLOT_MOVING_AVERAGE = True
MOVING_AVERAGE_WINDOWS = [250]

BASELINE_MOVING_AVERAGE_WINDOW = 5000

PLOT_ONLINE_MOVING_AVERAGE_COMPARISON = True
PLOT_ONLINE_CUMULATIVE_SUM_OF_MOVING_AVERAGE_COMPARISON = True
PLOT_RAW_MA_AND_CUMULATIVE_SUM_SECOND_AXIS = True

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

APPLY_AUTO_Y_LIMITS = True

Y_LIMIT_PADDING_FRACTION = 0.08


# ============================================================
# COLOR SETTINGS
# ============================================================

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
    # {
    #     "title": "Primitive Variables - Min",
    #     "filenames": [
    #         "prim_interp_min.txt",
    #         "prim_proj_min.txt",
    #     ],
    #     "selected_vars": None,
    #     "logy": False,
    # },
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
    # {
    #     "title": "Conservative Variables - Min",
    #     "filenames": [
    #         "cons_interp_min.txt",
    #         "cons_proj_min.txt",
    #     ],
    #     "selected_vars": None,
    #     "logy": False,
    # },
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
    """
    Compute a CFD-style causal moving average.

    Before a full moving-average window is available, this computes
    a running average from the first sample through the current sample.

    Once enough samples are available, this computes a trailing moving
    average using only the current sample and previous samples.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window <= 0:
        raise ValueError("Moving-average window must be positive.")

    y_avg = np.full(len(y), np.nan)

    cumulative_sum = 0.0

    for i in range(len(y)):
        cumulative_sum += y[i]

        if i >= window:
            cumulative_sum -= y[i - window]
            y_avg[i] = cumulative_sum / window
        else:
            y_avg[i] = cumulative_sum / (i + 1)

    return x, y_avg


def compute_moving_average_ignore_nan(x, y, window):
    """
    Compute a CFD-style causal moving average while ignoring NaN values.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window <= 0:
        raise ValueError("Moving-average window must be positive.")

    y_avg = np.full(len(y), np.nan)

    cumulative_sum = 0.0
    cumulative_count = 0

    for i in range(len(y)):
        if np.isfinite(y[i]):
            cumulative_sum += y[i]
            cumulative_count += 1

        if i >= window:
            old_value = y[i - window]

            if np.isfinite(old_value):
                cumulative_sum -= old_value
                cumulative_count -= 1

        if cumulative_count > 0:
            y_avg[i] = cumulative_sum / cumulative_count
        else:
            y_avg[i] = np.nan

    return x, y_avg


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


def compute_online_moving_average(x, y, window):
    return compute_moving_average(x, y, window)


def compute_online_moving_average_comparison_cumulative_sum(
    x,
    y,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
):
    """
    Compute the CFD-style comparison:

        cumsum(MA-user-window - MA-baseline-window)

    Both moving averages use only information available at each iteration.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_short, y_short_ma = compute_online_moving_average(
        x,
        y,
        moving_average_window,
    )

    x_long, y_long_ma = compute_online_moving_average(
        x,
        y,
        baseline_moving_average_window,
    )

    moving_average_difference = np.full(len(y), np.nan)
    cumulative_moving_average_difference = np.full(len(y), np.nan)

    cumulative_sum = 0.0

    for i in range(len(y)):
        if not np.isfinite(y_short_ma[i]):
            continue

        if not np.isfinite(y_long_ma[i]):
            continue

        difference = y_short_ma[i] - y_long_ma[i]

        moving_average_difference[i] = difference

        cumulative_sum += difference

        cumulative_moving_average_difference[i] = cumulative_sum

    return (
        x,
        y_short_ma,
        y_long_ma,
        moving_average_difference,
        cumulative_moving_average_difference,
    )


# ============================================================
# Y-AXIS LIMIT HELPERS
# ============================================================

def collect_line_y_data(line, logy=False):
    y = np.asarray(line.get_ydata(), dtype=float)
    y = y[np.isfinite(y)]

    if logy:
        y = y[y > 0.0]

    return y


def get_adaptive_outlier_fence_limits(y, logy=False):
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


def merge_twin_axis_legends(ax_left, ax_right, fontsize=10):
    left_lines, left_labels = ax_left.get_legend_handles_labels()
    right_lines, right_labels = ax_right.get_legend_handles_labels()

    ax_left.legend(
        left_lines + right_lines,
        left_labels + right_labels,
        fontsize=fontsize,
        frameon=True,
        loc="best",
    )


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
                    label=f"{filename.stem} CFD MA-{moving_average_window}",
                    zorder=5,
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
        figure_title += f" [CFD MA window = {moving_average_window}]"

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

                x_avg, y_avg = compute_moving_average(
                    x,
                    y,
                    moving_average_window,
                )

                slope_x = x_avg
                slope_y = compute_slope(x_avg, y_avg)
                source_label = f"CFD MA-{moving_average_window} slope"

            else:
                raise ValueError("slope_source must be either 'raw' or 'ma'.")

            if plot_magnitude:
                slope_y = np.abs(slope_y)
                line_label = f"{filename.stem} |{source_label}|"
                ma_label = (
                    f"{filename.stem} |{source_label}| "
                    f"CFD MA-{moving_average_window}"
                )

            elif plot_positive_only:
                slope_y = np.where(slope_y > 0.0, slope_y, np.nan)
                line_label = f"{filename.stem} positive {source_label}"
                ma_label = (
                    f"{filename.stem} positive {source_label} "
                    f"CFD MA-{moving_average_window}"
                )

            else:
                line_label = f"{filename.stem} {source_label}"
                ma_label = (
                    f"{filename.stem} {source_label} "
                    f"CFD MA-{moving_average_window}"
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
                f" Raw Slope CFD Moving Average Only "
                f"[MA window = {moving_average_window}]"
            )
        else:
            figure_title += " Slope [raw slope]"

    elif slope_source == "ma":
        if plot_magnitude:
            figure_title += (
                f" Slope Magnitude "
                f"[|CFD MA-{moving_average_window} slope|]"
            )
        else:
            figure_title += f" Slope [CFD MA-{moving_average_window} slope]"

    if plot_moving_average_of_slope and plot_slope_curve:
        figure_title += f" [slope CFD MA window = {moving_average_window}]"

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
                        f"CFD MA-{moving_average_window}"
                    ),
                    zorder=6,
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
        figure_title += f" [cumsum CFD MA window = {moving_average_window}]"

    finalize_figure(fig, axes, figure_title)

    return fig


# ============================================================
# ONLINE MOVING-AVERAGE COMPARISON PLOTS
# ============================================================

def plot_online_moving_average_comparison_data(
    group,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
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

            if moving_average_window is None:
                print(
                    f"Warning: moving_average_window is None for {filename}. "
                    "Skipping online moving-average comparison."
                )
                continue

            (
                x_comparison,
                y_short_ma,
                y_long_ma,
                moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=baseline_moving_average_window,
            )

            ax.plot(
                x_comparison,
                y_short_ma,
                color=primary_color,
                linewidth=2.4,
                alpha=0.95,
                label=f"{filename.stem} CFD MA-{moving_average_window}",
                zorder=5,
            )

            ax.plot(
                x_comparison,
                y_long_ma,
                color=secondary_color,
                linewidth=2.4,
                linestyle="--",
                alpha=0.95,
                label=f"{filename.stem} CFD MA-{baseline_moving_average_window}",
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

    figure_title = (
        f"{title} CFD-Style Moving Averages "
        f"[MA-{moving_average_window} and MA-{baseline_moving_average_window}]"
    )

    if logy:
        figure_title += " [log-y]"

    finalize_figure(fig, axes, figure_title)

    return fig


def plot_online_cumulative_moving_average_comparison_data(
    group,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
):
    loaded_data, variable_indices = load_group_data(group)

    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for file_idx, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            secondary_color = SECONDARY_COLORS[file_idx % len(SECONDARY_COLORS)]

            if moving_average_window is None:
                print(
                    f"Warning: moving_average_window is None for {filename}. "
                    "Skipping online cumulative moving-average comparison."
                )
                continue

            (
                x_comparison,
                y_short_ma,
                y_long_ma,
                moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=baseline_moving_average_window,
            )

            ax.plot(
                x_comparison,
                cumulative_moving_average_difference,
                color=secondary_color,
                linewidth=3.0,
                alpha=1.0,
                label=(
                    f"{filename.stem} cumsum "
                    f"(CFD MA-{moving_average_window} minus "
                    f"CFD MA-{baseline_moving_average_window})"
                ),
                zorder=6,
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
            ylabel=f"cumsum MA comparison var{var_index + 1}",
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            logy=False,
            keep_zero_visible=True,
        )

    figure_title = (
        f"{title} CFD-Style Cumulative Sum of Moving-Average Comparison "
        f"[MA-{moving_average_window} minus MA-{baseline_moving_average_window}]"
    )

    finalize_figure(fig, axes, figure_title)

    return fig


def plot_raw_moving_average_and_cumulative_sum_second_axis(
    group,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
):
    loaded_data, variable_indices = load_group_data(group)

    logy = group.get("logy", True)
    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax_left, var_index in zip(axes, variable_indices):
        ax_right = ax_left.twinx()

        for file_idx, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            primary_color = PRIMARY_COLORS[file_idx % len(PRIMARY_COLORS)]
            secondary_color = SECONDARY_COLORS[file_idx % len(SECONDARY_COLORS)]
            tertiary_color = TERTIARY_COLORS[file_idx % len(TERTIARY_COLORS)]

            if moving_average_window is None:
                print(
                    f"Warning: moving_average_window is None for {filename}. "
                    "Skipping raw/MA/cumsum second-axis plot."
                )
                continue

            (
                x_comparison,
                y_short_ma,
                y_long_ma,
                moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=baseline_moving_average_window,
            )

            ax_left.plot(
                x,
                y,
                color=primary_color,
                linewidth=1.0,
                alpha=0.55,
                label=f"{filename.stem} raw",
                zorder=3,
            )

            ax_left.plot(
                x_comparison,
                y_short_ma,
                color=secondary_color,
                linewidth=2.6,
                alpha=1.0,
                label=f"{filename.stem} CFD MA-{moving_average_window}",
                zorder=5,
            )

            ax_right.plot(
                x_comparison,
                cumulative_moving_average_difference,
                color=tertiary_color,
                linewidth=2.6,
                linestyle="-.",
                alpha=0.95,
                label=(
                    f"{filename.stem} cumsum "
                    f"(MA-{moving_average_window} - MA-{baseline_moving_average_window})"
                ),
                zorder=6,
            )

        ax_left.set_ylabel(f"var{var_index + 1}", fontsize=12)
        ax_right.set_ylabel(
            f"cumsum(MA-{moving_average_window} - MA-{baseline_moving_average_window})",
            fontsize=12,
        )

        if logy:
            ax_left.set_yscale("log")

        ax_left.set_axisbelow(True)

        if SHOW_GRID:
            ax_left.grid(
                True,
                which="both",
                linestyle="--",
                linewidth=0.7,
                alpha=0.6,
                zorder=0,
            )

        ax_left.tick_params(axis="both", labelsize=11)
        ax_right.tick_params(axis="y", labelsize=11)

        merge_twin_axis_legends(ax_left, ax_right, fontsize=9)

        apply_limits_if_enabled(
            ax_left,
            logy=logy,
            keep_zero_visible=False,
        )

        apply_limits_if_enabled(
            ax_right,
            logy=False,
            keep_zero_visible=True,
        )

    figure_title = (
        f"{title} Raw Data, CFD MA, and Cumulative MA Comparison "
        f"[MA-{moving_average_window} vs MA-{baseline_moving_average_window}]"
    )

    if logy:
        figure_title += " [left axis log-y]"

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

            if plot_moving_average and PLOT_ONLINE_MOVING_AVERAGE_COMPARISON:
                fig = plot_online_moving_average_comparison_data(
                    group,
                    moving_average_window=moving_average_window,
                    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
                )
                add_page_to_pdf(pdf, fig)

            if (
                plot_moving_average
                and PLOT_ONLINE_CUMULATIVE_SUM_OF_MOVING_AVERAGE_COMPARISON
            ):
                fig = plot_online_cumulative_moving_average_comparison_data(
                    group,
                    moving_average_window=moving_average_window,
                    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
                )
                add_page_to_pdf(pdf, fig)

            if (
                plot_moving_average
                and PLOT_RAW_MA_AND_CUMULATIVE_SUM_SECOND_AXIS
            ):
                fig = plot_raw_moving_average_and_cumulative_sum_second_axis(
                    group,
                    moving_average_window=moving_average_window,
                    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
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