#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# USER SETTINGS
# Edit FIGURE_GROUPS, then run:
#
#     python plot_txt_data.py
#
# ============================================================

DELIMITER = ","
SKIPROWS = 50
SHOW_GRID = True
SHOW_FIGURES = False

# Moving average settings
PLOT_MOVING_AVERAGE = True

# Multiple moving-average windows to calculate.
# Each window gets its own figures and PDF.
MOVING_AVERAGE_WINDOWS = [50, 100, 250]

# Slope settings
CALCULATE_SLOPE = True

# Plot slope of raw data on its own slope figure
PLOT_SLOPE_OF_RAW = True

# Plot slope of moving-average data on its own slope figure
PLOT_SLOPE_OF_MOVING_AVERAGE = True

# Usually keep slope plots linear because slopes can be negative.
SLOPE_LOGY = False

# Robust y-axis limit settings.
# These prevent extreme outliers from stretching the y-axis so much
# that the main data becomes difficult to parse.
APPLY_ROBUST_Y_LIMITS = True

# Percentile bounds used for y-axis limits.
# Example: (1.0, 99.0) ignores the lowest 1% and highest 1% of values
# when choosing the displayed y-axis range.
Y_LIMIT_PERCENTILES = (1.0, 99.0)

# Fractional padding added around the percentile-based y-limits.
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
    "cyan",      # complements blue
    "red",       # complements orange
    "lime",      # complements green
    "magenta",   # complements purple
]

# Directory where saved figures will go.
SAVE_DIR = "figures"

# Combined PDF filename base.
# The moving-average window will be added automatically.
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
# FUNCTIONS
# ============================================================

def load_data(filename, delimiter=",", skiprows=0):
    """
    Load data from a text file.

    Expected format:
        iteration, var1, var2, ..., varn
    """
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

    iterations = data[:, 0]
    variables = data[:, 1:]

    return iterations, variables


def compute_moving_average(x, y, window):
    """
    Compute a simple moving average.

    Uses mode='valid', so the moving-average curve is shorter than
    the original curve. The x-values are shifted to stay centered.
    """
    kernel = np.ones(window) / window
    y_avg = np.convolve(y, kernel, mode="valid")

    start = window // 2
    end = start + len(y_avg)
    x_avg = x[start:end]

    return x_avg, y_avg


def compute_slope(x, y):
    """
    Compute dy/dx using numpy.gradient.

    This returns a slope array with the same length as x and y.
    """
    slope = np.gradient(y, x)
    return slope


def add_filename_suffix(filename, suffix):
    """
    Add a suffix before the file extension.

    Example:
        prim_max_comparison.png + _MA_100
        -> prim_max_comparison_MA_100.png
    """
    filename = Path(filename)
    return str(filename.with_name(f"{filename.stem}{suffix}{filename.suffix}"))


def make_slope_output_file(output_file, slope_type, output_suffix=""):
    """
    Create a slope output filename.

    Example:
        prim_max_comparison.png, raw, _MA_100
        -> prim_max_comparison_raw_slope_MA_100.png

        prim_max_comparison.png, ma, _MA_100
        -> prim_max_comparison_ma_slope_MA_100.png
    """
    output_file = Path(output_file)

    return str(
        output_file.with_name(
            f"{output_file.stem}_{slope_type}_slope{output_suffix}{output_file.suffix}"
        )
    )


def collect_axis_y_data(ax, logy=False):
    """
    Collect y-data from all plotted lines on an axis.

    If logy=True, only positive values are kept because log-scale axes
    cannot display zero or negative values.
    """
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


def apply_robust_y_limits(
    ax,
    logy=False,
    percentiles=(1.0, 99.0),
    padding_fraction=0.08,
):
    """
    Apply percentile-based y-axis limits.

    This avoids extreme outliers dominating the visible y-range.

    For linear axes:
        ymin/ymax are based on the requested percentiles plus padding.

    For log axes:
        only positive values are considered, and padding is applied
        multiplicatively in log10 space.
    """
    y = collect_axis_y_data(ax, logy=logy)

    if len(y) == 0:
        return

    lower_percentile, upper_percentile = percentiles
    ymin, ymax = np.percentile(y, [lower_percentile, upper_percentile])

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return

    if ymin == ymax:
        if logy:
            if ymin <= 0.0:
                return
            ax.set_ylim(ymin * 0.9, ymax * 1.1)
        else:
            pad = abs(ymin) * 0.1
            if pad == 0.0:
                pad = 1.0
            ax.set_ylim(ymin - pad, ymax + pad)
        return

    if logy:
        if ymin <= 0.0 or ymax <= 0.0:
            return

        log_ymin = np.log10(ymin)
        log_ymax = np.log10(ymax)
        log_range = log_ymax - log_ymin

        if log_range <= 0.0:
            return

        log_pad = padding_fraction * log_range

        ax.set_ylim(
            10.0 ** (log_ymin - log_pad),
            10.0 ** (log_ymax + log_pad),
        )

    else:
        y_range = ymax - ymin
        pad = padding_fraction * y_range

        ax.set_ylim(ymin - pad, ymax + pad)


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
):
    """
    Plot one figure containing one or more files.

    Each variable is plotted in its own subplot.
    Files listed together are overlaid on the same axes.

    If enabled, this plots:
        raw data
        moving average of raw data
    """

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

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):

        for file_idx, (filename, iterations, variables) in enumerate(loaded_data):

            raw_color = FILE_COLORS[file_idx % len(FILE_COLORS)]
            ma_color = MOVING_AVERAGE_COLORS[file_idx % len(MOVING_AVERAGE_COLORS)]

            x = iterations
            y = variables[:, var_index]

            if plot_moving_average:

                # ----------------------------------------------------
                # Raw data
                # ----------------------------------------------------
                ax.plot(
                    x,
                    y,
                    color=raw_color,
                    linewidth=1.0,
                    alpha=0.70,
                    label=filename.stem,
                    zorder=2,
                )

                # ----------------------------------------------------
                # Moving average of raw data
                # ----------------------------------------------------
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

            else:

                ax.plot(
                    x,
                    y,
                    color=raw_color,
                    linewidth=2.0,
                    label=filename.stem,
                    zorder=3,
                )

        format_axis(
            ax=ax,
            var_index=var_index,
            logy=logy,
            show_grid=show_grid,
        )

        if APPLY_ROBUST_Y_LIMITS:
            apply_robust_y_limits(
                ax=ax,
                logy=logy,
                percentiles=Y_LIMIT_PERCENTILES,
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
):
    """
    Plot one slope figure for one or more files.

    Each variable is plotted in its own subplot.
    Files listed together are overlaid on the same axes.

    slope_type options:
        "raw" -> plot slope of raw data
        "ma"  -> plot slope of moving-average data
    """

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

    fig, axes = create_figure(variable_indices)

    for ax, var_index in zip(axes, variable_indices):

        for file_idx, (filename, iterations, variables) in enumerate(loaded_data):

            raw_color = FILE_COLORS[file_idx % len(FILE_COLORS)]
            ma_color = MOVING_AVERAGE_COLORS[file_idx % len(MOVING_AVERAGE_COLORS)]

            x = iterations
            y = variables[:, var_index]

            # --------------------------------------------------------
            # Slope of raw data
            # --------------------------------------------------------
            if slope_type == "raw":

                raw_slope = compute_slope(x, y)

                ax.plot(
                    x,
                    raw_slope,
                    color=raw_color,
                    linewidth=1.5,
                    alpha=0.65,
                    label=f"{filename.stem} raw slope",
                    zorder=3,
                )

            # --------------------------------------------------------
            # Slope of moving-average data
            # --------------------------------------------------------
            elif slope_type == "ma":

                if plot_moving_average:

                    if moving_average_window <= len(y):

                        x_avg, y_avg = compute_moving_average(
                            x,
                            y,
                            moving_average_window,
                        )

                        ma_slope = compute_slope(x_avg, y_avg)

                        ax.plot(
                            x_avg,
                            ma_slope,
                            color=ma_color,
                            linewidth=2.5,
                            alpha=1.0,
                            label=f"{filename.stem} MA-{moving_average_window} slope",
                            zorder=5,
                        )

                    else:
                        print(
                            f"Warning: moving_average_window={moving_average_window} "
                            f"is larger than data length={len(y)} for {filename}. "
                            f"Skipping slope of moving average."
                        )

            else:
                raise ValueError("slope_type must be either 'raw' or 'ma'.")

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

        ax.set_ylabel(f"d(var{var_index + 1})/dIteration", fontsize=12)

        if APPLY_ROBUST_Y_LIMITS:
            apply_robust_y_limits(
                ax=ax,
                logy=slope_logy,
                percentiles=Y_LIMIT_PERCENTILES,
                padding_fraction=Y_LIMIT_PADDING_FRACTION,
            )

    axes[-1].set_xlabel("Iteration", fontsize=12)

    if slope_type == "raw":
        figure_title = f"{title} Slope [raw slope]"
    elif slope_type == "ma":
        figure_title = f"{title} Slope [MA-{moving_average_window} slope]"

    fig.suptitle(figure_title, fontsize=15, fontweight="bold")

    if save_plot:
        save_figure(fig, output_file, save_dir=save_dir)

    return fig, axes


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
    slope_logy=False,
):
    """
    Plot multiple figures in one run.

    Each entry in figure_groups creates one main figure.

    If calculate_slope=True, corresponding raw-slope and/or
    moving-average-slope figures are also created immediately after
    each main figure.

    Memory-saving behavior:
        Each figure is written to the PDF immediately and then closed.
        Figures are not stored in a list.
    """

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
            # Main raw + moving-average figure
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
            # Raw slope figure
            # --------------------------------------------------------
            if (
                group.get("calculate_slope", calculate_slope)
                and group.get("plot_slope_of_raw", plot_slope_of_raw)
            ):

                raw_slope_output_file = make_slope_output_file(
                    group.get("output_file", "plot.png"),
                    slope_type="raw",
                    output_suffix=output_suffix,
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
                )

                pdf.savefig(raw_slope_fig, bbox_inches="tight")

                if not show_figures:
                    plt.close(raw_slope_fig)

            # --------------------------------------------------------
            # Moving-average slope figure
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
                )

                pdf.savefig(ma_slope_fig, bbox_inches="tight")

                if not show_figures:
                    plt.close(ma_slope_fig)

    print(f"Saved combined PDF to: {pdf_path}")

    if show_figures:
        plt.show()
    else:
        plt.close("all")


def create_figure(variable_indices):
    """
    Create a figure with one subplot per variable.
    """
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
    """
    Apply common formatting to one subplot.
    """
    ax.set_ylabel(f"var{var_index + 1}", fontsize=12)

    if logy:
        ax.set_yscale("log")

    # Keep grid behind plotted data.
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
    """
    Build the output path.

    If output_file is only a filename, it is saved inside save_dir.
    If output_file already includes a directory, that path is used directly.
    """

    output_file = Path(output_file)

    if output_file.parent == Path("."):
        output_path = Path(save_dir) / output_file
    else:
        output_path = output_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path


def save_figure(fig, output_file, save_dir="figures"):
    """
    Save one figure as an image.
    """

    output_path = get_output_path(output_file, save_dir=save_dir)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")


def main():

    if PLOT_MOVING_AVERAGE:

        for window in MOVING_AVERAGE_WINDOWS:

            print("")
            print("============================================================")
            print(f"Creating plots for moving-average window = {window}")
            print("============================================================")

            # Save each moving-average window in its own folder.
            window_save_dir = Path(SAVE_DIR) / f"MA_{window}"

            # Add moving-average window to filenames.
            output_suffix = f"_MA_{window}"

            # Add moving-average window to PDF filename.
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
                slope_logy=SLOPE_LOGY,
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
            slope_logy=SLOPE_LOGY,
        )


if __name__ == "__main__":
    main()