#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename, delimiter=",", skiprows=0):
    """
    Load numerical data from a text file.

    Expected format:
        iteration var1 var2 ... varn

    Parameters
    ----------
    filename : str or Path
        Path to the input file.
    delimiter : str or None
        Delimiter used in the file. None means any whitespace.
    skiprows : int
        Number of rows to skip at the top of the file.

    Returns
    -------
    iterations : ndarray
        First column of the data.
    variables : ndarray
        Remaining columns of the data.
    """
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=skiprows)

    if data.ndim == 1:
        raise ValueError(
            f"File '{filename}' appears to contain only one row. "
            "At least two rows are recommended for plotting."
        )

    if data.shape[1] < 2:
        raise ValueError(
            f"File '{filename}' must contain at least two columns: "
            "iteration and one variable."
        )

    iterations = data[:, 0]
    variables = data[:, 1:]

    return iterations, variables


def plot_files(
    filenames,
    delimiter=None,
    skiprows=0,
    selected_vars=None,
    output=None,
    title=None,
    show_grid=True,
):
    """
    Plot one or more data files.

    If multiple files are provided, variables with the same column index
    are plotted together on the same subplot.
    """

    loaded_data = []

    max_num_vars = None

    for filename in filenames:
        iterations, variables = load_data(filename, delimiter, skiprows)
        num_vars = variables.shape[1]

        if max_num_vars is None:
            max_num_vars = num_vars
        else:
            max_num_vars = max(max_num_vars, num_vars)

        loaded_data.append((Path(filename), iterations, variables))

    if selected_vars is None:
        variable_indices = list(range(max_num_vars))
    else:
        # User enters variables as 1-based: var1, var2, ...
        variable_indices = [v - 1 for v in selected_vars]

    num_plots = len(variable_indices)

    fig, axes = plt.subplots(
        num_plots,
        1,
        figsize=(10, 3.2 * num_plots),
        sharex=True,
        constrained_layout=True,
    )

    if num_plots == 1:
        axes = [axes]

    for ax, var_index in zip(axes, variable_indices):
        for filename, iterations, variables in loaded_data:
            if var_index >= variables.shape[1]:
                print(
                    f"Warning: {filename} does not contain var{var_index + 1}. "
                    "Skipping this variable for that file."
                )
                continue

            label = f"{filename.stem} - var{var_index + 1}"

            ax.plot(
                iterations,
                variables[:, var_index],
                linewidth=2.0,
                label=label,
            )

        ax.set_ylabel(f"var{var_index + 1}", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)

        if show_grid:
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

        ax.legend(fontsize=10, frameon=True)

    axes[-1].set_xlabel("Iteration", fontsize=12)

    if title is None:
        if len(filenames) == 1:
            title = f"Data Plot: {Path(filenames[0]).name}"
        else:
            title = "Data Comparison"

    fig.suptitle(title, fontsize=15, fontweight="bold")

    if output is not None:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot iteration-based data from one or more .txt files."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Input .txt data file or files.",
    )

    parser.add_argument(
        "--delimiter",
        default=None,
        help=(
            "Column delimiter. Default is whitespace. "
            "Use ',' for comma-separated files."
        ),
    )

    parser.add_argument(
        "--skiprows",
        type=int,
        default=0,
        help="Number of header rows to skip. Default is 0.",
    )

    parser.add_argument(
        "--vars",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Variables to plot using 1-based numbering. "
            "Example: --vars 1 3 plots var1 and var3. "
            "Default is to plot all variables."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image filename, e.g. plot.png or plot.pdf.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )

    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable plot grid.",
    )

    args = parser.parse_args()

    plot_files(
        filenames=args.files,
        delimiter=args.delimiter,
        skiprows=args.skiprows,
        selected_vars=args.vars,
        output=args.output,
        title=args.title,
        show_grid=not args.no_grid,
    )


if __name__ == "__main__":
    main()
