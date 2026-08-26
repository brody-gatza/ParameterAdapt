import argparse
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")  # HPC-safe non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_common_data(fom_dir, rom_dir):
    """
    Load FOM and ROM files for all common iterations.

    Expected filename:
        {iteration}iteration_prim.npy

    Expected shape of each file:
        (variable, cell)

    Returned data shape:
        (time, variable, cell)
    """

    pattern = re.compile(r"^(\d+)iteration_prim\.npy$")

    def find_files(directory):
        files = {}

        for filename in os.listdir(directory):
            match = pattern.match(filename)

            if match:
                iteration = int(match.group(1))
                files[iteration] = os.path.join(
                    directory,
                    filename
                )

        return files

    if not os.path.isdir(fom_dir):
        raise NotADirectoryError(
            f"FOM directory does not exist:\n{fom_dir}"
        )

    if not os.path.isdir(rom_dir):
        raise NotADirectoryError(
            f"ROM directory does not exist:\n{rom_dir}"
        )

    fom_files = find_files(fom_dir)
    rom_files = find_files(rom_dir)

    iterations = np.array(
        sorted(set(fom_files) & set(rom_files)),
        dtype=int
    )

    if len(iterations) == 0:
        raise RuntimeError(
            "No common FOM and ROM iteration files were found.\n"
            f"FOM directory: {fom_dir}\n"
            f"ROM directory: {rom_dir}"
        )

    fom_data = np.array([
        np.load(fom_files[iteration])
        for iteration in iterations
    ])

    rom_data = np.array([
        np.load(rom_files[iteration])
        for iteration in iterations
    ])

    if fom_data.shape != rom_data.shape:
        raise ValueError(
            "FOM and ROM data shapes do not match.\n"
            f"FOM shape: {fom_data.shape}\n"
            f"ROM shape: {rom_data.shape}"
        )

    if fom_data.ndim != 3:
        raise ValueError(
            "Expected assembled data shape "
            "(time, variable, cell).\n"
            f"Received shape: {fom_data.shape}"
        )

    print("============================================================")
    print("Loaded FOM and ROM data")
    print(f"Common iterations: {len(iterations)}")
    print(f"First iteration:   {iterations[0]}")
    print(f"Last iteration:    {iterations[-1]}")
    print(f"Data shape:        {fom_data.shape}")
    print("============================================================")

    return fom_data, rom_data, iterations


def compute_histories(
    fom_data,
    rom_data,
    tolerance=1.0e-14
):
    """
    Compute the relative L2 error and Pearson correlation at every
    time step for every field.

    FOM is treated as the truth value.

    Relative L2 error:

        ||ROM - FOM||_2 / ||FOM||_2

    Metrics are computed across the spatial cells.

    Returned array shapes:
        relative_l2: (variable, time)
        correlation: (variable, time)
    """

    number_of_times = fom_data.shape[0]
    number_of_variables = fom_data.shape[1]

    relative_l2 = np.full(
        (number_of_variables, number_of_times),
        np.nan
    )

    correlation = np.full(
        (number_of_variables, number_of_times),
        np.nan
    )

    for time_index in range(number_of_times):

        for variable_index in range(number_of_variables):

            fom = fom_data[
                time_index,
                variable_index,
                :
            ]

            rom = rom_data[
                time_index,
                variable_index,
                :
            ]

            valid = np.isfinite(fom) & np.isfinite(rom)

            if not np.any(valid):
                continue

            fom_valid = fom[valid]
            rom_valid = rom[valid]

            # Relative L2 error
            fom_norm = np.linalg.norm(fom_valid)

            if fom_norm > tolerance:
                relative_l2[
                    variable_index,
                    time_index
                ] = (
                    np.linalg.norm(rom_valid - fom_valid)
                    / fom_norm
                )

            # Pearson correlation
            if (
                len(fom_valid) >= 2
                and np.std(fom_valid) > tolerance
                and np.std(rom_valid) > tolerance
            ):
                correlation[
                    variable_index,
                    time_index
                ] = np.corrcoef(
                    fom_valid,
                    rom_valid
                )[0, 1]

    print("Computed error and correlation histories.")

    return relative_l2, correlation


def plot_histories_to_pdf(
    iterations,
    relative_l2,
    correlation,
    field_names,
    pdf_filename
):
    """
    Save all variable histories to one multipage PDF.

    Each variable receives one PDF page with:
        Left y-axis:  relative L2 error in percent
        Right y-axis: Pearson correlation
        x-axis:       iteration

    Major and minor ticks are shown on the x-axis.
    """

    number_of_variables = relative_l2.shape[0]

    if len(field_names) != number_of_variables:
        print(
            "Warning: field-name count does not match "
            "the number of variables."
        )
        print("Using generic field names.")

        field_names = [
            f"Field {index}"
            for index in range(number_of_variables)
        ]

    output_directory = os.path.dirname(
        os.path.abspath(pdf_filename)
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    print("============================================================")
    print("Creating metric plots")
    print(f"Number of variables: {number_of_variables}")
    print(f"Output PDF: {os.path.abspath(pdf_filename)}")
    print("============================================================")

    with PdfPages(pdf_filename) as pdf:

        for variable_index, field_name in enumerate(field_names):

            print(
                f"Plotting variable "
                f"{variable_index + 1}/{number_of_variables}: "
                f"{field_name}"
            )

            # Create one plot with two y-axes.
            fig, error_axis = plt.subplots(
                figsize=(10, 6)
            )

            correlation_axis = error_axis.twinx()

            # ----------------------------------------------------
            # Relative L2 error on the left axis
            # ----------------------------------------------------

            error_line, = error_axis.plot(
                iterations,
                100.0 * relative_l2[variable_index],
                color="tab:red",
                linestyle="-",
                linewidth=1.5,
                label=r"Relative $L_2$ error"
            )

            error_axis.set_xlabel(
                "Iteration"
            )

            error_axis.set_ylabel(
                r"Relative $L_2$ error [%]",
                color="tab:red"
            )

            error_axis.tick_params(
                axis="y",
                labelcolor="tab:red"
            )

            # ----------------------------------------------------
            # Major and minor x-axis ticks
            # ----------------------------------------------------

            error_axis.minorticks_on()

            error_axis.tick_params(
                axis="x",
                which="major",
                length=6
            )

            error_axis.tick_params(
                axis="x",
                which="minor",
                length=3
            )

            # Major grid lines on both axes.
            error_axis.grid(
                which="major",
                axis="both",
                linestyle="-",
                alpha=0.35
            )

            # Minor grid lines on the x-axis only.
            error_axis.grid(
                which="minor",
                axis="x",
                linestyle=":",
                alpha=0.20
            )

            # ----------------------------------------------------
            # Pearson correlation on the right axis
            # ----------------------------------------------------

            correlation_line, = correlation_axis.plot(
                iterations,
                correlation[variable_index],
                color="tab:blue",
                linestyle="-",
                linewidth=1.5,
                label="Pearson correlation"
            )

            correlation_axis.set_ylabel(
                "Pearson correlation",
                color="tab:blue"
            )

            correlation_axis.tick_params(
                axis="y",
                labelcolor="tab:blue"
            )

            correlation_axis.set_ylim(
                -1.05,
                1.05
            )

            correlation_axis.axhline(
                y=1.0,
                color="tab:blue",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5
            )

            correlation_axis.axhline(
                y=0.0,
                color="tab:blue",
                linestyle=":",
                linewidth=0.8,
                alpha=0.5
            )

            # ----------------------------------------------------
            # Title and legend
            # ----------------------------------------------------

            error_axis.set_title(
                field_name
            )

            error_axis.legend(
                handles=[
                    error_line,
                    correlation_line
                ],
                loc="best"
            )

            fig.tight_layout()

            # Add the current figure as one page in the PDF.
            pdf.savefig(
                fig,
                bbox_inches="tight"
            )

            plt.close(fig)

            print(
                f"Finished variable "
                f"{variable_index + 1}/{number_of_variables}: "
                f"{field_name}"
            )

    print("============================================================")
    print("Finished creating all metric plots.")
    print("Saved multipage PDF:")
    print(f"  {os.path.abspath(pdf_filename)}")
    print("============================================================")


def parse_arguments():
    """
    Read command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute relative L2 error and Pearson correlation "
            "histories between FOM and ROM result files."
        )
    )

    parser.add_argument(
        "fom_dir",
        help=(
            "Directory containing the FOM "
            "{iteration}iteration_prim.npy files."
        )
    )

    parser.add_argument(
        "rom_dir",
        help=(
            "Directory containing the ROM "
            "{iteration}iteration_prim.npy files."
        )
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output PDF filename. By default, "
            "FOM_ROM_metrics.pdf is saved in the ROM directory."
        )
    )

    return parser.parse_args()


def main():
    """
    Main program.
    """

    args = parse_arguments()

    fom_dir = os.path.abspath(
        os.path.expanduser(args.fom_dir)
    )

    rom_dir = os.path.abspath(
        os.path.expanduser(args.rom_dir)
    )

    if args.output is None:
        pdf_filename = os.path.join(
            rom_dir,
            "FOM_ROM_metrics.pdf"
        )
    else:
        pdf_filename = os.path.abspath(
            os.path.expanduser(args.output)
        )

    field_names = [
        "Density",
        "Velocity",
        "Pressure",
        "Temperature",
        "Reactant",
        "Heat release rate"
    ]

    print("============================================================")
    print("FOM directory:")
    print(f"  {fom_dir}")
    print("ROM directory:")
    print(f"  {rom_dir}")
    print("Output PDF:")
    print(f"  {pdf_filename}")
    print("============================================================")

    fom_data, rom_data, iterations = load_common_data(
        fom_dir=fom_dir,
        rom_dir=rom_dir
    )

    relative_l2, correlation = compute_histories(
        fom_data=fom_data,
        rom_data=rom_data
    )

    plot_histories_to_pdf(
        iterations=iterations,
        relative_l2=relative_l2,
        correlation=correlation,
        field_names=field_names,
        pdf_filename=pdf_filename
    )


if __name__ == "__main__":
    main()