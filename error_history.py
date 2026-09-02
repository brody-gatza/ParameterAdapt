import argparse
import os
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # HPC-safe non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages




# Uniform computational-cell volume.
# Change this value if the computational-cell volume changes.
# CELL_VOLUME = 0.000144
# Reported values are already multiplied by volume
CELL_VOLUME = 1


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
    pdf_filename,
    pdf=None
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

    with (PdfPages(pdf_filename) if pdf is None else nullcontext(pdf)) as pdf:

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
                1.00,
                0.00
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






def load_common_conservative_data(fom_dir, rom_dir, iterations):
    """
    Load FOM and ROM conservative-state files for the supplied iterations.

    Expected filename:
        {iteration}iteration_cons.npy

    Expected shape of each file:
        (conservative_variable, cell)

    Returned data shape:
        (time, conservative_variable, cell)
    """
    pattern = re.compile(r"^(\d+)iteration_cons\.npy$")

    def find_files(directory):
        files = {}
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                iteration = int(match.group(1))
                files[iteration] = os.path.join(directory, filename)
        return files

    fom_files = find_files(fom_dir)
    rom_files = find_files(rom_dir)

    missing_fom = [
        int(iteration)
        for iteration in iterations
        if int(iteration) not in fom_files
    ]
    missing_rom = [
        int(iteration)
        for iteration in iterations
        if int(iteration) not in rom_files
    ]

    if missing_fom or missing_rom:
        message = [
            "Conservative files are not available for every common "
            "primitive-data iteration."
        ]
        if missing_fom:
            message.append(
                "Missing FOM conservative iterations: "
                + ", ".join(str(value) for value in missing_fom[:20])
            )
            if len(missing_fom) > 20:
                message.append(
                    f"Also missing {len(missing_fom) - 20} additional "
                    "FOM conservative files."
                )
        if missing_rom:
            message.append(
                "Missing ROM conservative iterations: "
                + ", ".join(str(value) for value in missing_rom[:20])
            )
            if len(missing_rom) > 20:
                message.append(
                    f"Also missing {len(missing_rom) - 20} additional "
                    "ROM conservative files."
                )
        raise FileNotFoundError("\n".join(message))

    fom_conservative_data = np.array([
        np.load(fom_files[int(iteration)])
        for iteration in iterations
    ])
    rom_conservative_data = np.array([
        np.load(rom_files[int(iteration)])
        for iteration in iterations
    ])

    if fom_conservative_data.shape != rom_conservative_data.shape:
        raise ValueError(
            "FOM and ROM conservative data shapes do not match.\n"
            f"FOM conservative shape: {fom_conservative_data.shape}\n"
            f"ROM conservative shape: {rom_conservative_data.shape}"
        )

    if fom_conservative_data.ndim != 3:
        raise ValueError(
            "Expected assembled conservative data shape "
            "(time, conservative_variable, cell).\n"
            f"Received shape: {fom_conservative_data.shape}"
        )

    if fom_conservative_data.shape[0] != len(iterations):
        raise ValueError(
            "The number of conservative time steps does not match the "
            "number of supplied iterations.\n"
            f"Conservative time steps: {fom_conservative_data.shape[0]}\n"
            f"Iterations: {len(iterations)}"
        )

    if fom_conservative_data.shape[1] < 4:
        raise ValueError(
            "The conservative data must contain at least four fields "
            "in this order: density, momentum density, total energy "
            "density, and species density.\n"
            f"Received {fom_conservative_data.shape[1]} fields."
        )

    print("============================================================")
    print("Loaded FOM and ROM conservative data")
    print(f"Common iterations:       {len(iterations)}")
    print(f"First iteration:         {iterations[0]}")
    print(f"Last iteration:          {iterations[-1]}")
    print(f"Conservative data shape: {fom_conservative_data.shape}")
    print("============================================================")

    return fom_conservative_data, rom_conservative_data


def compute_conservative_integral_histories(
    fom_conservative_data,
    rom_conservative_data,
):
    """
    Compute domain-integrated conservative quantities.

    Fixed conservative field order:
        index 0: density
        index 1: momentum density
        index 2: total energy density
        index 3: species density

    Integration is performed across all cells at each time step.
    """
    if not np.isfinite(CELL_VOLUME) or CELL_VOLUME <= 0.0:
        raise ValueError(
            "CELL_VOLUME must be a positive finite value. "
            f"Received: {CELL_VOLUME}"
        )

    if fom_conservative_data.shape != rom_conservative_data.shape:
        raise ValueError(
            "FOM and ROM conservative data shapes do not match.\n"
            f"FOM shape: {fom_conservative_data.shape}\n"
            f"ROM shape: {rom_conservative_data.shape}"
        )

    conservative_fields = {
        "mass": 0,
        "momentum": 1,
        "energy": 2,
        "species": 3,
    }

    def integrate(data, data_name):
        if data.ndim != 3 or data.shape[1] < 4:
            raise ValueError(
                f"{data_name} conservative data must have shape "
                "(time, at_least_four_variables, cell). "
                f"Received shape: {data.shape}"
            )

        histories = {}
        for quantity_name, variable_index in conservative_fields.items():
            field = np.asarray(data[:, variable_index, :], dtype=np.float64)

            if not np.all(np.isfinite(field)):
                bad_times = np.flatnonzero(
                    np.any(~np.isfinite(field), axis=1)
                )
                raise ValueError(
                    f"{data_name} {quantity_name} contains non-finite "
                    f"values at {len(bad_times)} time steps. "
                    f"First affected time index: {int(bad_times[0])}."
                )

            histories[quantity_name] = CELL_VOLUME * np.sum(
                field,
                axis=1,
                dtype=np.float64,
            )

        return histories

    fom_integrals = integrate(fom_conservative_data, "FOM")
    rom_integrals = integrate(rom_conservative_data, "ROM")

    number_of_cells = fom_conservative_data.shape[2]
    represented_volume = CELL_VOLUME * number_of_cells

    print("============================================================")
    print("Computed domain-integrated conservative quantities")
    print(f"Cell volume:              {CELL_VOLUME:.16g}")
    print(f"Number of cells:          {number_of_cells}")
    print(f"Total represented volume: {represented_volume:.16g}")
    print("Conservative variable order:")
    print("  0: density")
    print("  1: momentum density")
    print("  2: total energy density")
    print("  3: species density")
    print("============================================================")

    return fom_integrals, rom_integrals


def plot_conservative_integral_histories_to_pdf(
    iterations,
    fom_integrals,
    rom_integrals,
    pdf,
    tolerance=1.0e-14,
):
    """
    Append integrated mass, momentum, energy, and species plots.

    The left axis shows the FOM and ROM integrated quantities.
    The right axis shows the relative error in percent:

        100 * abs(ROM - FOM) / abs(FOM)

    If the absolute FOM value is less than or equal to tolerance, the
    relative error is set to NaN to avoid division by zero or by a value
    too close to zero.
    """
    plot_definitions = (
        ("mass", "Domain-integrated mass", "Total mass"),
        ("momentum", "Domain-integrated momentum", "Total momentum"),
        ("energy", "Domain-integrated energy", "Total energy"),
        ("species", "Domain-integrated species mass", "Total species mass"),
    )

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "The conserved-quantity error tolerance must be a finite, "
            f"nonnegative value. Received: {tolerance}"
        )

    iterations = np.asarray(iterations, dtype=float)

    print("============================================================")
    print("Appending domain-integrated conservative histories")
    print("The right axis shows relative FOM-ROM error in percent")
    print("============================================================")

    with nullcontext(pdf) as pdf:
        for key, title, y_label in plot_definitions:
            fom_values = np.asarray(fom_integrals[key], dtype=float)
            rom_values = np.asarray(rom_integrals[key], dtype=float)

            if fom_values.shape != iterations.shape:
                raise ValueError(
                    f"FOM {key} history shape {fom_values.shape} does not "
                    f"match iteration shape {iterations.shape}."
                )

            if rom_values.shape != iterations.shape:
                raise ValueError(
                    f"ROM {key} history shape {rom_values.shape} does not "
                    f"match iteration shape {iterations.shape}."
                )

            valid = (
                np.isfinite(fom_values)
                & np.isfinite(rom_values)
                & (np.abs(fom_values) > tolerance)
            )

            relative_error_percent = np.full(
                fom_values.shape,
                np.nan,
                dtype=float,
            )

            relative_error_percent[valid] = (
                100.0
                * np.abs(rom_values[valid] - fom_values[valid])
                / np.abs(fom_values[valid])
            )

            invalid_count = np.count_nonzero(~valid)
            if invalid_count:
                print(
                    f"Warning: {invalid_count} {key} error values were set "
                    "to NaN because the FOM value was non-finite or too "
                    "close to zero, or the ROM value was non-finite."
                )

            fig, quantity_axis = plt.subplots(figsize=(10, 6))
            error_axis = quantity_axis.twinx()

            fom_line, = quantity_axis.plot(
                iterations,
                fom_values,
                color="black",
                linestyle="-",
                linewidth=1.8,
                label="FOM",
            )

            rom_line, = quantity_axis.plot(
                iterations,
                rom_values,
                color="tab:orange",
                linestyle="--",
                linewidth=1.8,
                label="ROM",
            )

            error_line, = error_axis.plot(
                iterations,
                relative_error_percent,
                color="tab:red",
                linestyle=":",
                linewidth=1.8,
                label="Relative error",
            )

            quantity_axis.set_xlabel("Iteration")
            quantity_axis.set_ylabel(y_label)
            error_axis.set_ylabel("Relative error [%]", color="tab:red")
            error_axis.tick_params(axis="y", labelcolor="tab:red")

            quantity_axis.set_title(title)
            quantity_axis.minorticks_on()
            quantity_axis.tick_params(axis="x", which="major", length=6)
            quantity_axis.tick_params(axis="x", which="minor", length=3)

            quantity_axis.grid(
                which="major",
                axis="both",
                linestyle="-",
                alpha=0.35,
            )

            quantity_axis.grid(
                which="minor",
                axis="x",
                linestyle=":",
                alpha=0.20,
            )

            quantity_axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-3, 3),
                useMathText=True,
            )

            error_axis.ticklabel_format(
                axis="y",
                style="sci",
                scilimits=(-3, 3),
                useMathText=True,
            )

            quantity_axis.legend(
                handles=[fom_line, rom_line, error_line],
                loc="best",
            )

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            finite_error = relative_error_percent[
                np.isfinite(relative_error_percent)
            ]

            if len(finite_error):
                print(
                    f"Added: {title}; maximum relative error: "
                    f"{np.max(finite_error):.6e}%"
                )
            else:
                print(
                    f"Added: {title}; no finite relative-error values"
                )

    print(
        "Finished appending domain-integrated conservative histories."
    )



ERROR_SIGNAL_GROUPS = (
    ("Primitive Variables - Maximum Error", "prim_interp_max.txt", "primitive"),
    ("Conservative Variables - Maximum Error", "cons_interp_max.txt", "conservative"),
    ("Primitive Variables - Average Error", "prim_interp_avg.txt", "primitive"),
    ("Conservative Variables - Average Error", "cons_interp_avg.txt", "conservative"),
)


def compute_moving_average_ignore_nan(values, window):
    """Compute a trailing moving average, ignoring non-finite samples."""
    values = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")

    finite = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite.astype(int))))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    window_sums = sums[end] - sums[start]
    window_counts = counts[end] - counts[start]

    average = np.full(len(values), np.nan)
    valid = window_counts > 0
    average[valid] = window_sums[valid] / window_counts[valid]
    return average


def load_error_signal_file(filename, delimiter=","):
    """Load iteration from column zero and error signals from later columns."""
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Error-signal file does not exist:\n{filename}")

    data = np.loadtxt(filename, delimiter=delimiter, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected iteration plus at least one signal column in {filename}."
        )

    iterations = np.asarray(data[:, 0], dtype=float)
    signals = np.asarray(data[:, 1:], dtype=float)
    order = np.argsort(iterations, kind="stable")
    iterations = iterations[order]
    signals = signals[order]

    if np.any(np.diff(iterations) == 0.0):
        raise ValueError(f"Duplicate iterations were found in {filename}.")
    return iterations, signals


def interpolate_metric_to_iterations(source_iterations, values, target_iterations):
    """Interpolate a metric to signal iterations without extrapolation."""
    source_iterations = np.asarray(source_iterations, dtype=float)
    values = np.asarray(values, dtype=float)
    target_iterations = np.asarray(target_iterations, dtype=float)
    valid = np.isfinite(source_iterations) & np.isfinite(values)

    if np.count_nonzero(valid) == 0:
        return np.full(target_iterations.shape, np.nan)

    x = source_iterations[valid]
    y = values[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    x, indices = np.unique(x, return_index=True)
    y = y[indices]

    if len(x) == 1:
        result = np.full(target_iterations.shape, np.nan)
        result[np.isclose(target_iterations, x[0])] = y[0]
        return result
    return np.interp(target_iterations, x, y, left=np.nan, right=np.nan)


def get_error_signal_field_names(kind, count, primitive_field_names):
    if kind == "primitive" and len(primitive_field_names) == count:
        return list(primitive_field_names)
    prefix = "Primitive" if kind == "primitive" else "Conservative"
    return [f"{prefix} field {index + 1}" for index in range(count)]


def plot_error_signals_to_pdf(
    metric_iterations,
    relative_l2,
    correlation,
    primitive_field_names,
    error_directory,
    pdf,
    short_window=1000,
    long_window=5000,
    delimiter=",",
):
    """Append raw signals, two moving averages, L2 error, and correlation."""
    error_directory = Path(error_directory)
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")

    available = []
    for title, basename, kind in ERROR_SIGNAL_GROUPS:
        filename = error_directory / basename
        if filename.is_file():
            available.append((title, filename, kind))
        else:
            print(f"Warning: missing error-signal file: {filename}")

    if not available:
        print("No error-signal pages were created.")
        return

    print("============================================================")
    print("Appending error-signal plots")
    print(f"Error directory: {error_directory.resolve()}")
    print(f"Moving-average windows: {short_window}, {long_window}")
    print("============================================================")

    with nullcontext(pdf) as pdf:
        for title, filename, kind in available:
            signal_iterations, signals = load_error_signal_file(
                filename, delimiter=delimiter
            )
            names = get_error_signal_field_names(
                kind, signals.shape[1], primitive_field_names
            )

            for variable_index, field_name in enumerate(names):
                raw = signals[:, variable_index]
                short_average = compute_moving_average_ignore_nan(raw, short_window)
                long_average = compute_moving_average_ignore_nan(raw, long_window)

                fig, signal_axis = plt.subplots(figsize=(12, 7))
                l2_axis = signal_axis.twinx()
                correlation_axis = signal_axis.twinx()
                correlation_axis.spines["right"].set_position(("axes", 1.13))
                correlation_axis.patch.set_visible(False)

                lines = []
                lines += signal_axis.plot(
                    signal_iterations, raw, color="0.55", linewidth=0.8,
                    alpha=0.55, label="Raw error signal", zorder=2
                )
                lines += signal_axis.plot(
                    signal_iterations, short_average, color="tab:orange",
                    linewidth=2.0, label=f"Moving average ({short_window})", zorder=4
                )
                lines += signal_axis.plot(
                    signal_iterations, long_average, color="tab:green",
                    linewidth=2.2, linestyle="-.",
                    label=f"Moving average ({long_window})", zorder=5
                )

                if variable_index < relative_l2.shape[0]:
                    aligned_l2 = interpolate_metric_to_iterations(
                        metric_iterations,
                        100.0 * relative_l2[variable_index],
                        signal_iterations,
                    )
                    aligned_correlation = interpolate_metric_to_iterations(
                        metric_iterations,
                        correlation[variable_index],
                        signal_iterations,
                    )
                    lines += l2_axis.plot(
                        signal_iterations, aligned_l2, color="tab:red",
                        linewidth=1.5, alpha=0.90,
                        label=r"Relative $L_2$ error [%]", zorder=7
                    )
                    lines += correlation_axis.plot(
                        signal_iterations, aligned_correlation, color="tab:blue",
                        linewidth=1.5, alpha=0.90,
                        label="Pearson correlation", zorder=8
                    )
                else:
                    print(
                        f"Warning: no FOM/ROM metric field {variable_index + 1} "
                        f"for {filename.name}."
                    )

                finite_raw = raw[np.isfinite(raw)]
                if len(finite_raw) and np.all(finite_raw > 0.0):
                    signal_axis.set_yscale("log")

                signal_axis.set_xlabel("Iteration")
                signal_axis.set_ylabel("Error signal")
                l2_axis.set_ylabel(r"Relative $L_2$ error [%]", color="tab:red")
                correlation_axis.set_ylabel(
                    "Pearson correlation", color="tab:blue", labelpad=12
                )
                l2_axis.tick_params(axis="y", labelcolor="tab:red")
                correlation_axis.tick_params(axis="y", labelcolor="tab:blue")
                correlation_axis.set_ylim(1.00, 0.00)
                signal_axis.minorticks_on()
                signal_axis.grid(which="major", axis="both", linestyle="-", alpha=0.30)
                signal_axis.grid(which="minor", axis="x", linestyle=":", alpha=0.18)
                correlation_axis.axhline(
                    1.0, color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.4
                )
                correlation_axis.axhline(
                    0.0, color="tab:blue", linestyle=":", linewidth=0.8, alpha=0.4
                )
                signal_axis.legend(
                    lines, [line.get_label() for line in lines],
                    loc="best", fontsize=9
                )
                signal_axis.set_title(f"{title}: {field_name}")
                fig.subplots_adjust(right=0.80)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print(f"Added: {title}: {field_name}")

    print("Finished appending error-signal plots.")


def plot_fom_rom_conservation_errors_to_pdf(
    fom_error_directory,
    rom_error_directory,
    pdf,
    short_window=1000,
    long_window=5000,
    delimiter=",",
):
    """
    Append ROM conservation-error plots with short and long moving averages.

    Four pages are created for each conservative variable:
        1. line plot with logarithmic y-axis
        2. line plot with linear y-axis
        3. scatter plot with logarithmic y-axis
        4. scatter plot with linear y-axis

    FOM conservation-error data is still loaded and validated against the ROM
    variable count, but the FOM plotting block is retained as comments and no
    FOM conservation-error pages are added to the PDF.
    """
    fom_error_directory = Path(fom_error_directory)
    rom_error_directory = Path(rom_error_directory)

    if short_window <= 0 or long_window <= 0:
        raise ValueError("Moving-average windows must be positive integers.")

    fom_error_iterations, fom_error_values = load_error_signal_file(
        fom_error_directory / "error_cons.txt",
        delimiter=delimiter,
    )
    rom_error_iterations, rom_error_values = load_error_signal_file(
        rom_error_directory / "error_cons.txt",
        delimiter=delimiter,
    )

    fom_percent_iterations, fom_percent_values = load_error_signal_file(
        fom_error_directory / "error_cons_perct.txt",
        delimiter=delimiter,
    )
    rom_percent_iterations, rom_percent_values = load_error_signal_file(
        rom_error_directory / "error_cons_perct.txt",
        delimiter=delimiter,
    )

    if fom_error_values.shape[1] != rom_error_values.shape[1]:
        raise ValueError(
            "FOM and ROM conservation-error files contain different numbers "
            "of conservative variables: "
            f"{fom_error_values.shape[1]} versus {rom_error_values.shape[1]}"
        )

    if fom_percent_values.shape[1] != rom_percent_values.shape[1]:
        raise ValueError(
            "FOM and ROM percent conservation-error files contain different "
            "numbers of conservative variables: "
            f"{fom_percent_values.shape[1]} versus "
            f"{rom_percent_values.shape[1]}"
        )

    if rom_error_values.shape[1] != rom_percent_values.shape[1]:
        raise ValueError(
            "ROM absolute and percent conservation-error files contain "
            "different numbers of conservative variables: "
            f"{rom_error_values.shape[1]} versus "
            f"{rom_percent_values.shape[1]}"
        )

    # Conservation errors are plotted as magnitudes. This also ensures that
    # valid nonzero samples can be displayed on logarithmic axes.
    fom_error_values = np.abs(fom_error_values)
    rom_error_values = np.abs(rom_error_values)
    fom_percent_values = np.abs(fom_percent_values)
    rom_percent_values = np.abs(rom_percent_values)

    default_names = ["Mass", "Momentum", "Energy", "Species"]
    number_of_variables = rom_error_values.shape[1]
    field_names = (
        default_names
        if number_of_variables == len(default_names)
        else [
            f"Conservative field {index + 1}"
            for index in range(number_of_variables)
        ]
    )

    plot_variants = (
        ("Scatter, logarithmic y-axis", "scatter", True),
        ("Scatter, linear y-axis", "scatter", False),
    )

    def add_rom_page(
        field_name,
        values,
        short_average,
        long_average,
        plot_label,
        plot_kind,
        use_log_y,
        value_iterations=None,
        y_label="Conservation error magnitude",
        title_prefix="ROM Conservation Error History",
    ):
        """Add one ROM raw-error scatter plot and two moving averages."""
        if value_iterations is None:
            value_iterations = rom_error_iterations
        fig, value_axis = plt.subplots(figsize=(12, 7))

        if use_log_y:
            raw_plot_values = np.where(
                np.isfinite(values) & (values > 0.0), values, np.nan
            )
            short_plot_values = np.where(
                np.isfinite(short_average) & (short_average > 0.0),
                short_average,
                np.nan,
            )
            long_plot_values = np.where(
                np.isfinite(long_average) & (long_average > 0.0),
                long_average,
                np.nan,
            )
        else:
            raw_plot_values = values
            short_plot_values = short_average
            long_plot_values = long_average

        if plot_kind == "scatter":
            raw_artist = value_axis.scatter(
                value_iterations,
                raw_plot_values,
                color="tab:red",
                s=10,
                alpha=0.50,
                label="ROM value",
                zorder=2,
            )
        else:
            raw_artist, = value_axis.plot(
                value_iterations,
                raw_plot_values,
                color="tab:red",
                linestyle="-",
                linewidth=1.0,
                alpha=0.60,
                label="ROM value",
                zorder=2,
            )

        short_line, = value_axis.plot(
            value_iterations,
            short_plot_values,
            color="darkblue",
            linestyle="-",
            linewidth=2.0,
            label=f"Moving average ({short_window})",
            zorder=4,
        )
        long_line, = value_axis.plot(
            value_iterations,
            long_plot_values,
            color="tab:green",
            linestyle="-.",
            linewidth=2.2,
            label=f"Moving average ({long_window})",
            zorder=5,
        )

        value_axis.set_xlabel("Iteration")
        value_axis.set_ylabel(y_label)
        value_axis.set_title(
            f"ROM Conservation Error History: {field_name}\n{plot_label}"
        )
        if use_log_y:
            value_axis.set_yscale("log")

        value_axis.minorticks_on()
        value_axis.tick_params(axis="x", which="major", length=6)
        value_axis.tick_params(axis="x", which="minor", length=3)
        value_axis.grid(
            which="major",
            axis="both",
            linestyle="-",
            alpha=0.35,
        )
        value_axis.grid(
            which="minor",
            axis="both",
            linestyle=":",
            alpha=0.20,
        )
        value_axis.legend(
            handles=[raw_artist, short_line, long_line],
            loc="best",
            fontsize=9,
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print("============================================================")
    print("Appending ROM conservation-error histories")
    print("FOM conservation-error plotting is commented out")
    print(f"ROM variables: {number_of_variables}")
    print(f"Pages per ROM variable: {len(plot_variants)}")
    print(f"Moving-average windows: {short_window}, {long_window}")
    print("============================================================")

    # FOM conservation-error plots are intentionally disabled.
    # The former FOM plotting loop would be placed here. FOM data continues
    # to be loaded above so the input files and conservative-variable counts
    # are still checked.
    #
    # for variable_index, field_name in enumerate(field_names):
    #     add_model_page(
    #         model_name="FOM",
    #         field_name=field_name,
    #         value_iterations=fom_error_iterations,
    #         values=fom_error_values[:, variable_index],
    #         value_color="darkblue",
    #     )

    for variable_index, field_name in enumerate(field_names):
        values = rom_error_values[:, variable_index]
        short_average = compute_moving_average_ignore_nan(
            values, short_window
        )
        long_average = compute_moving_average_ignore_nan(
            values, long_window
        )

        for plot_label, plot_kind, use_log_y in plot_variants:
            add_rom_page(
                field_name=field_name,
                values=values,
                short_average=short_average,
                long_average=long_average,
                plot_label=plot_label,
                plot_kind=plot_kind,
                use_log_y=use_log_y,
            )
            print(
                f"Added ROM conservation-error page "
                f"{variable_index + 1}/{number_of_variables}: "
                f"{field_name}, {plot_label}"
            )

        percent_values = rom_percent_values[:, variable_index]
        percent_short_average = compute_moving_average_ignore_nan(
            percent_values, short_window
        )
        percent_long_average = compute_moving_average_ignore_nan(
            percent_values, long_window
        )

        for plot_label, plot_kind, use_log_y in plot_variants:
            add_rom_page(
                field_name=field_name,
                values=percent_values,
                short_average=percent_short_average,
                long_average=percent_long_average,
                plot_label=plot_label,
                plot_kind=plot_kind,
                use_log_y=use_log_y,
                value_iterations=rom_percent_iterations,
                y_label="Conservation error [%]",
                title_prefix="ROM Percent Conservation Error History",
            )
            print(
                f"Added ROM percent conservation-error page "
                f"{variable_index + 1}/{number_of_variables}: "
                f"{field_name}, {plot_label}"
            )

    print("Finished appending ROM conservation-error histories.")


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


    parser.add_argument(
        "--error-short-window", type=int, default=1000,
        help="Short error-signal moving-average window (default: 1000)."
    )
    parser.add_argument(
        "--error-long-window", type=int, default=5000,
        help="Long error-signal moving-average window (default: 5000)."
    )
    parser.add_argument(
        "--error-delimiter", default=",",
        help="Error-signal file delimiter (default: comma)."
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


    fom_conservative_data, rom_conservative_data = (
        load_common_conservative_data(
            fom_dir=fom_dir,
            rom_dir=rom_dir,
            iterations=iterations,
        )
    )

    fom_integrals, rom_integrals = (
        compute_conservative_integral_histories(
            fom_conservative_data=fom_conservative_data,
            rom_conservative_data=rom_conservative_data,
        )
    )

    # Each error directory is a sibling of its corresponding results directory.
    fom_error_directory = Path(fom_dir) / "error"
    rom_error_directory = Path(rom_dir).parent / "error"
    error_directory = rom_error_directory

    # Open the output PDF exactly once so every figure is written to the same file.
    with PdfPages(pdf_filename) as pdf:
        plot_histories_to_pdf(
            iterations=iterations,
            relative_l2=relative_l2,
            correlation=correlation,
            field_names=field_names,
            pdf_filename=pdf_filename,
            pdf=pdf,
        )


        plot_conservative_integral_histories_to_pdf(
            iterations=iterations,
            fom_integrals=fom_integrals,
            rom_integrals=rom_integrals,
            pdf=pdf,
        )

        plot_fom_rom_conservation_errors_to_pdf(
            fom_error_directory=fom_error_directory,
            rom_error_directory=rom_error_directory,
            pdf=pdf,
            short_window=args.error_short_window,
            long_window=args.error_long_window,
            delimiter=args.error_delimiter,
        )

        plot_error_signals_to_pdf(
            metric_iterations=iterations,
            relative_l2=relative_l2,
            correlation=correlation,
            primitive_field_names=field_names,
            error_directory=error_directory,
            pdf=pdf,
            short_window=args.error_short_window,
            long_window=args.error_long_window,
            delimiter=args.error_delimiter,
        )


if __name__ == "__main__":
    main()