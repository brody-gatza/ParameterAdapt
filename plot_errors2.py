#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

# Each dictionary below creates one figure.
#
# filenames:
#     List of files to plot together on that figure.
#
# selected_vars:
#     Variables are 1-based.
#     Example: [1, 3] plots var1 and var3.
#     Use None to plot all variables.
#
# logy:
#     True  -> logarithmic y-axis
#     False -> linear y-axis
#
# save_plot:
#     True  -> save figure as image
#     False -> only display figure
#
# output_file:
#     Filename used if save_plot is True.
#
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
        "logy": True,
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
        "logy": True,
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
        "logy": True,
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
        "logy": True,
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


def plot_files(
    filenames,
    selected_vars=None,
    delimiter=",",
    skiprows=0,
    logy=True,
    title="Data Comparison",
    save_plot=False,
    output_file="plot.png",
    show_grid=True,
):
    """
    Plot one figure containing one or more files.

    Each variable is plotted in its own subplot.
    Files listed together are overlaid on the same axes.
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
        for filename, iterations, variables in loaded_data:
            x = iterations
            y = variables[:, var_index]

            if logy:
                x, y = keep_positive_y_values(x, y)

            ax.plot(
                x,
                y,
                linewidth=2.0,
                label=filename.stem,
            )

        format_axis(
            ax=ax,
            var_index=var_index,
            logy=logy,
            show_grid=show_grid,
        )

    axes[-1].set_xlabel("Iteration", fontsize=12)

    if logy:
        figure_title = f"{title} [log-y]"
    else:
        figure_title = title

    fig.suptitle(figure_title, fontsize=15, fontweight="bold")

    if save_plot:
        save_figure(fig, output_file)

    return fig, axes


def plot_figure_groups(
    figure_groups,
    delimiter=",",
    skiprows=0,
    show_grid=True,
):
    """
    Plot multiple figures in one run.

    Each entry in figure_groups creates one figure.
    """

    for group in figure_groups:
        plot_files(
            filenames=group["filenames"],
            selected_vars=group.get("selected_vars", None),
            delimiter=delimiter,
            skiprows=skiprows,
            logy=group.get("logy", True),
            title=group.get("title", "Data Comparison"),
            save_plot=group.get("save_plot", False),
            output_file=group.get("output_file", "plot.png"),
            show_grid=show_grid,
        )

    plt.show()


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

    if show_grid:
        ax.grid(
            True,
            which="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.6,
        )

    ax.legend(fontsize=10, frameon=True)
    ax.tick_params(axis="both", labelsize=11)


def keep_positive_y_values(x, y):
    """
    Remove non-positive y-values for log-y plotting.
    """
    positive_mask = y > 0
    return x[positive_mask], y[positive_mask]


def save_figure(fig, output_file):
    """
    Save figure as an image.
    """
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_file}")


def main():
    plot_figure_groups(
        figure_groups=FIGURE_GROUPS,
        delimiter=DELIMITER,
        skiprows=SKIPROWS,
        show_grid=SHOW_GRID,
    )


if __name__ == "__main__":
    main()
