#!/usr/bin/env python3

from pathlib import Path

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

SHOW_GRID = True
SHOW_FIGURES = False

PLOT_MOVING_AVERAGE = True
MOVING_AVERAGE_WINDOWS = [250]

BASELINE_MOVING_AVERAGE_WINDOW = 5000

PLOT_ONLINE_CUMULATIVE_SUM_OF_MOVING_AVERAGE_COMPARISON = True
PLOT_SLOPE_OF_CUMULATIVE_SUM_OF_MOVING_AVERAGE_COMPARISON = True
PLOT_RAW_MA_AND_CUMULATIVE_SUM_SECOND_AXIS = True

PLOT_RUNNING_AVERAGE = True

CALCULATE_SLOPE = True

PLOT_SLOPE_OF_RAW = False
PLOT_SLOPE_OF_MOVING_AVERAGE = False
PLOT_SLOPE_MAGNITUDE = False
PLOT_MOVING_AVERAGE_OF_SLOPE = False
PLOT_RAW_SLOPE_MOVING_AVERAGE_ONLY = False
PLOT_POSITIVE_RAW_SLOPES_ONLY = False

PLOT_CUMULATIVE_SUM_OF_ERROR_SLOPE = True
PLOT_MOVING_AVERAGE_OF_CUMULATIVE_SUM = True

SLOPE_LOGY = False
SLOPE_MAGNITUDE_LOGY = False
CUMULATIVE_SLOPE_LOGY = False


# ============================================================
# ITERATION LIMIT SETTINGS
# ============================================================

MIN_ITERATION_TO_INCLUDE = None
MAX_ITERATION_TO_INCLUDE = None


# ============================================================
# PLOT FILE SIZE / SIMPLIFICATION SETTINGS
# ============================================================

PLOT_EVERY_NTH_POINT = 1


# ============================================================
# Y-AXIS LIMIT SETTINGS
# ============================================================

APPLY_AUTO_Y_LIMITS = True
APPLY_AUTO_Y_LIMITS_TO_SLOPE_PLOTS = False

Y_LIMIT_PADDING_FRACTION = 0.08

AUTO_Y_LIMITS_IGNORE_EXTREME_OUTLIERS = True
Y_LIMIT_LOG_MIN_POSITIVE_VALUE = None


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
]


# ============================================================
# DATA FUNCTIONS
# ============================================================

def apply_iteration_limits(iterations, variables, filename):
    iterations = np.asarray(iterations, dtype=float)
    variables = np.asarray(variables, dtype=float)

    keep_mask = np.ones_like(iterations, dtype=bool)

    if MIN_ITERATION_TO_INCLUDE is not None:
        keep_mask &= iterations >= MIN_ITERATION_TO_INCLUDE

    if MAX_ITERATION_TO_INCLUDE is not None:
        keep_mask &= iterations <= MAX_ITERATION_TO_INCLUDE

    iterations = iterations[keep_mask]
    variables = variables[keep_mask, :]

    if len(iterations) == 0:
        raise ValueError(
            f"No data remains in {filename} after applying iteration limits: "
            f"MIN_ITERATION_TO_INCLUDE = {MIN_ITERATION_TO_INCLUDE}, "
            f"MAX_ITERATION_TO_INCLUDE = {MAX_ITERATION_TO_INCLUDE}."
        )

    return iterations, variables


def load_data(filename, delimiter=DELIMITER, skiprows=SKIPROWS):
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

    iterations = data[:, 0]
    variables = data[:, 1:]

    iterations, variables = apply_iteration_limits(
        iterations,
        variables,
        filename,
    )

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


def downsample_for_plotting(x, y, every_nth_point=PLOT_EVERY_NTH_POINT):
    if every_nth_point is None:
        return x, y

    if every_nth_point <= 1:
        return x, y

    return x[::every_nth_point], y[::every_nth_point]


def compute_moving_average(x, y, window):
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


def compute_cfd_style_slope_per_iteration(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slope = np.full(len(y), np.nan)

    for i in range(1, len(y)):
        dx = x[i] - x[i - 1]

        if not np.isfinite(dx):
            continue

        if dx == 0.0:
            continue

        if not np.isfinite(y[i]):
            continue

        if not np.isfinite(y[i - 1]):
            continue

        slope[i] = (y[i] - y[i - 1]) / dx

    return x, slope


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    _, y_short_ma = compute_online_moving_average(
        x,
        y,
        moving_average_window,
    )

    _, y_long_ma = compute_online_moving_average(
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

def get_automatic_endpoint_trim_count(num_points):
    if num_points < 20:
        return 0

    trim_count = int(np.ceil(0.02 * num_points))
    trim_count = min(trim_count, 100)

    max_safe_trim = max(0, (num_points - 10) // 2)
    trim_count = min(trim_count, max_safe_trim)

    return trim_count


def automatically_trim_line_endpoints(y):
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        return y

    trim_count = get_automatic_endpoint_trim_count(len(y))

    if trim_count <= 0:
        return y

    if 2 * trim_count >= len(y):
        return y

    return y[trim_count:-trim_count]


def robust_center_and_scale(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return None, None

    center = np.median(y)
    absolute_deviation = np.abs(y - center)
    mad = np.median(absolute_deviation)

    if not np.isfinite(center):
        return None, None

    if not np.isfinite(mad):
        return None, None

    robust_sigma = 1.4826 * mad

    return center, robust_sigma


def get_percentile_limits_automatic(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return None

    n = len(y)

    if n < 20:
        lower_percentile = 0.0
        upper_percentile = 100.0
    elif n < 100:
        lower_percentile = 2.0
        upper_percentile = 98.0
    elif n < 1000:
        lower_percentile = 1.0
        upper_percentile = 99.0
    else:
        lower_percentile = 0.5
        upper_percentile = 99.5

    y_min = np.percentile(y, lower_percentile)
    y_max = np.percentile(y, upper_percentile)

    if not np.isfinite(y_min):
        return None

    if not np.isfinite(y_max):
        return None

    if y_min >= y_max:
        return None

    return y_min, y_max


def get_robust_automatic_y_limits(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return None

    if len(y) < 4:
        y_min = np.min(y)
        y_max = np.max(y)

        if y_min == y_max:
            pad = abs(y_min) * 0.1

            if pad == 0.0:
                pad = 1.0

            return y_min - pad, y_max + pad

        return y_min, y_max

    center, robust_sigma = robust_center_and_scale(y)

    if center is not None and robust_sigma is not None and robust_sigma > 0.0:
        modified_z_score_limit = 10.0

        lower_fence = center - modified_z_score_limit * robust_sigma
        upper_fence = center + modified_z_score_limit * robust_sigma

        visible_y = y[
            (y >= lower_fence)
            & (y <= upper_fence)
        ]

        minimum_fraction_to_keep = 0.50
        minimum_points_to_keep = min(10, len(y))

        enough_points_kept = len(visible_y) >= minimum_points_to_keep
        enough_fraction_kept = len(visible_y) >= minimum_fraction_to_keep * len(y)

        if enough_points_kept and enough_fraction_kept:
            y_min = np.min(visible_y)
            y_max = np.max(visible_y)

            if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                return y_min, y_max

    percentile_limits = get_percentile_limits_automatic(y)

    if percentile_limits is not None:
        return percentile_limits

    y_min = np.min(y)
    y_max = np.max(y)

    if not np.isfinite(y_min):
        return None

    if not np.isfinite(y_max):
        return None

    if y_min == y_max:
        pad = abs(y_min) * 0.1

        if pad == 0.0:
            pad = 1.0

        return y_min - pad, y_max + pad

    if y_min >= y_max:
        return None

    return y_min, y_max


def pad_y_limits(
    visible_min,
    visible_max,
    logy=False,
    padding_fraction=Y_LIMIT_PADDING_FRACTION,
    keep_zero_visible=False,
):
    if not np.isfinite(visible_min):
        return None

    if not np.isfinite(visible_max):
        return None

    if visible_min >= visible_max:
        return None

    if logy:
        if visible_min <= 0.0:
            return None

        if visible_max <= 0.0:
            return None

        log_min = np.log10(visible_min)
        log_max = np.log10(visible_max)

        log_range = log_max - log_min

        if log_range <= 0.0:
            return None

        log_pad = padding_fraction * log_range

        final_min = 10.0 ** (log_min - log_pad)
        final_max = 10.0 ** (log_max + log_pad)

    else:
        y_range = visible_max - visible_min

        if y_range <= 0.0:
            return None

        pad = padding_fraction * y_range

        final_min = visible_min - pad
        final_max = visible_max + pad

        if keep_zero_visible:
            if final_min > 0.0:
                final_min = 0.0
            elif final_max < 0.0:
                final_max = 0.0

    if not np.isfinite(final_min):
        return None

    if not np.isfinite(final_max):
        return None

    if final_min >= final_max:
        return None

    return final_min, final_max


def apply_adaptive_y_limits(
    ax,
    logy=False,
    padding_fraction=Y_LIMIT_PADDING_FRACTION,
    keep_zero_visible=False,
):
    if not AUTO_Y_LIMITS_IGNORE_EXTREME_OUTLIERS:
        return

    all_working_y_values = []

    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]

        if len(y) == 0:
            continue

        if logy:
            y = y[y > 0.0]

            if Y_LIMIT_LOG_MIN_POSITIVE_VALUE is not None:
                y = y[y >= Y_LIMIT_LOG_MIN_POSITIVE_VALUE]

            if len(y) == 0:
                continue

            working_y = np.log10(y)
        else:
            working_y = y.copy()

        working_y = working_y[np.isfinite(working_y)]

        if len(working_y) == 0:
            continue

        working_y = automatically_trim_line_endpoints(working_y)

        if len(working_y) > 0:
            all_working_y_values.append(working_y)

    if not all_working_y_values:
        return

    combined_working_y = np.concatenate(all_working_y_values)
    combined_working_y = combined_working_y[np.isfinite(combined_working_y)]

    if len(combined_working_y) == 0:
        return

    robust_limits = get_robust_automatic_y_limits(combined_working_y)

    if robust_limits is None:
        return

    visible_min_working, visible_max_working = robust_limits

    if not np.isfinite(visible_min_working):
        return

    if not np.isfinite(visible_max_working):
        return

    if visible_min_working >= visible_max_working:
        return

    if logy:
        visible_min = 10.0 ** visible_min_working
        visible_max = 10.0 ** visible_max_working

        if visible_min <= 0.0:
            return

        if visible_max <= 0.0:
            return

        if visible_min >= visible_max:
            return
    else:
        visible_min = visible_min_working
        visible_max = visible_max_working

    padded_limits = pad_y_limits(
        visible_min,
        visible_max,
        logy=logy,
        padding_fraction=padding_fraction,
        keep_zero_visible=keep_zero_visible,
    )

    if padded_limits is None:
        return

    final_min, final_max = padded_limits

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

def get_plot_color(dataset_index, color_role="primary"):
    """
    Select colors by dataset identity and plotted data role.

    dataset_index:
        0 for the first file/dataset,
        1 for the second file/dataset,
        etc.

    color_role:
        "primary"   -> original/base curve for that dataset
        "secondary" -> first derived/smoothed curve for that dataset
        "tertiary"  -> second derived/smoothed curve for that dataset
    """

    if color_role == "primary":
        color_list = PRIMARY_COLORS
    elif color_role == "secondary":
        color_list = SECONDARY_COLORS
    elif color_role == "tertiary":
        color_list = TERTIARY_COLORS
    else:
        raise ValueError(
            "color_role must be 'primary', 'secondary', or 'tertiary'."
        )

    return color_list[dataset_index % len(color_list)]


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
        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            raw_linewidth = 1.0 if plot_moving_average else 2.0
            raw_alpha = 0.70 if plot_moving_average else 0.85

            x_plot, y_plot = downsample_for_plotting(x, y)

            ax.plot(
                x_plot,
                y_plot,
                color=get_plot_color(dataset_index, "primary"),
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

                x_avg_plot, y_avg_plot = downsample_for_plotting(x_avg, y_avg)

                ax.plot(
                    x_avg_plot,
                    y_avg_plot,
                    color=get_plot_color(dataset_index, "secondary"),
                    linewidth=3.0,
                    alpha=1.0,
                    label=f"{filename.stem} CFD MA-{moving_average_window}",
                    zorder=5,
                )

            if plot_running_average:
                x_running, y_running = compute_running_average(x, y)

                x_running_plot, y_running_plot = downsample_for_plotting(
                    x_running,
                    y_running,
                )

                ax.plot(
                    x_running_plot,
                    y_running_plot,
                    color=get_plot_color(dataset_index, "tertiary"),
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
        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

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
                slope_x_plot, slope_y_plot = downsample_for_plotting(
                    slope_x,
                    slope_y,
                )

                ax.plot(
                    slope_x_plot,
                    slope_y_plot,
                    color=get_plot_color(dataset_index, "primary"),
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

                x_slope_avg_plot, slope_avg_plot = downsample_for_plotting(
                    x_slope_avg,
                    slope_avg,
                )

                ax.plot(
                    x_slope_avg_plot,
                    slope_avg_plot,
                    color=get_plot_color(dataset_index, "secondary"),
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

        if APPLY_AUTO_Y_LIMITS_TO_SLOPE_PLOTS:
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
        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            raw_slope = compute_slope(x, y)
            cumulative_raw_slope = compute_cumulative_sum(raw_slope)

            x_plot, cumulative_raw_slope_plot = downsample_for_plotting(
                x,
                cumulative_raw_slope,
            )

            ax.plot(
                x_plot,
                cumulative_raw_slope_plot,
                color=get_plot_color(dataset_index, "primary"),
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

                x_avg_plot, cumulative_avg_plot = downsample_for_plotting(
                    x_avg,
                    cumulative_avg,
                )

                ax.plot(
                    x_avg_plot,
                    cumulative_avg_plot,
                    color=get_plot_color(dataset_index, "secondary"),
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

        if APPLY_AUTO_Y_LIMITS_TO_SLOPE_PLOTS:
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

def plot_online_cumulative_moving_average_comparison_data(
    group,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
):
    loaded_data, variable_indices = load_group_data(group)

    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

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

            x_plot, cumsum_plot = downsample_for_plotting(
                x_comparison,
                cumulative_moving_average_difference,
            )

            ax.plot(
                x_plot,
                cumsum_plot,
                color=get_plot_color(dataset_index, "secondary"),
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


def plot_slope_of_cumulative_moving_average_comparison_data(
    group,
    moving_average_window,
    baseline_moving_average_window=BASELINE_MOVING_AVERAGE_WINDOW,
):
    loaded_data, variable_indices = load_group_data(group)

    title = group.get("title", "Data Comparison")

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):
        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

            if moving_average_window is None:
                print(
                    f"Warning: moving_average_window is None for {filename}. "
                    "Skipping slope of cumulative moving-average comparison."
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

            slope_x, slope_y = compute_cfd_style_slope_per_iteration(
                x_comparison,
                cumulative_moving_average_difference,
            )

            x_slope_plot, slope_y_plot = downsample_for_plotting(
                slope_x,
                slope_y,
            )

            ax.plot(
                x_slope_plot,
                slope_y_plot,
                color=get_plot_color(dataset_index, "primary"),
                linewidth=1.2,
                alpha=0.55,
                label=(
                    f"{filename.stem} slope of cumsum "
                    f"(MA-{moving_average_window} - MA-{baseline_moving_average_window})"
                ),
                zorder=3,
            )

            x_slope_avg, slope_avg = compute_moving_average_ignore_nan(
                slope_x,
                slope_y,
                moving_average_window,
            )

            x_slope_avg_plot, slope_avg_plot = downsample_for_plotting(
                x_slope_avg,
                slope_avg,
            )

            ax.plot(
                x_slope_avg_plot,
                slope_avg_plot,
                color=get_plot_color(dataset_index, "secondary"),
                linewidth=3.0,
                alpha=1.0,
                label=(
                    f"{filename.stem} slope of cumsum "
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
            ylabel=f"d cumsum MA comparison var{var_index + 1}/dIteration",
            logy=False,
        )

        if APPLY_AUTO_Y_LIMITS_TO_SLOPE_PLOTS:
            apply_limits_if_enabled(
                ax,
                logy=False,
                keep_zero_visible=True,
            )

    figure_title = (
        f"{title} CFD-Style Slope of Cumulative Moving-Average Comparison "
        f"[MA-{moving_average_window} minus MA-{baseline_moving_average_window}] "
        f"[slope MA window = {moving_average_window}]"
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

        for dataset_index, item in enumerate(loaded_data):
            filename = item["filename"]
            x = item["iterations"]
            y = item["variables"][:, var_index]

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

            x_raw_plot, y_raw_plot = downsample_for_plotting(x, y)

            x_short_plot, y_short_plot = downsample_for_plotting(
                x_comparison,
                y_short_ma,
            )

            x_cumsum_plot, cumsum_plot = downsample_for_plotting(
                x_comparison,
                cumulative_moving_average_difference,
            )

            ax_left.plot(
                x_raw_plot,
                y_raw_plot,
                color=get_plot_color(dataset_index, "primary"),
                linewidth=1.0,
                alpha=0.55,
                label=f"{filename.stem} raw",
                zorder=3,
            )

            ax_left.plot(
                x_short_plot,
                y_short_plot,
                color=get_plot_color(dataset_index, "secondary"),
                linewidth=2.6,
                alpha=1.0,
                label=f"{filename.stem} CFD MA-{moving_average_window}",
                zorder=5,
            )

            ax_right.plot(
                x_cumsum_plot,
                cumsum_plot,
                color=get_plot_color(dataset_index, "tertiary"),
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
            f"cumsum(MA-{moving_average_window} - {baseline_moving_average_window})",
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

def format_iteration_value_for_filename(value):
    if value is None:
        return None

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def get_iteration_range_suffix():
    if MIN_ITERATION_TO_INCLUDE is None and MAX_ITERATION_TO_INCLUDE is None:
        return ""

    start_text = format_iteration_value_for_filename(MIN_ITERATION_TO_INCLUDE)
    end_text = format_iteration_value_for_filename(MAX_ITERATION_TO_INCLUDE)

    if start_text is None:
        start_text = "start"

    if end_text is None:
        end_text = "end"

    return f"_iter_{start_text}_to_{end_text}"


def make_pdf_name(base_name, moving_average_window=None):
    base_name = Path(base_name)

    iteration_range_suffix = get_iteration_range_suffix()

    suffix_parts = []

    if iteration_range_suffix:
        suffix_parts.append(iteration_range_suffix)

    if moving_average_window is not None:
        suffix_parts.append(f"_MA_{moving_average_window}")

    if not suffix_parts:
        return base_name

    combined_suffix = "".join(suffix_parts)

    return base_name.with_name(
        f"{base_name.stem}{combined_suffix}{base_name.suffix}"
    )


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
                and PLOT_SLOPE_OF_CUMULATIVE_SUM_OF_MOVING_AVERAGE_COMPARISON
            ):
                fig = plot_slope_of_cumulative_moving_average_comparison_data(
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
    if MIN_ITERATION_TO_INCLUDE is None:
        print("Minimum iteration: using first available iteration")
    else:
        print(f"Minimum iteration: {MIN_ITERATION_TO_INCLUDE}")

    if MAX_ITERATION_TO_INCLUDE is None:
        print("Maximum iteration: using final available iteration")
    else:
        print(f"Maximum iteration: {MAX_ITERATION_TO_INCLUDE}")

    print(f"Output directory: {OUTPUT_DIR}")
    print("Existing files in the output directory will not be deleted.")

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
        pdf_path = OUTPUT_DIR / make_pdf_name(
            PDF_OUTPUT_FILE,
            moving_average_window=None,
        )

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