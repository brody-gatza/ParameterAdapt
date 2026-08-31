#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


NEW_FUNCTION = r'''def plot_fom_rom_conservation_errors_to_pdf(
    fom_error_directory,
    rom_error_directory,
    pdf,
    short_window=1000,
    long_window=5000,
    delimiter=",",
):
    """
    Append ROM conservation-error scatter plots.

    For each conservative variable, four ROM pages are created:

        1. Absolute conservation error, logarithmic y-axis
        2. Absolute conservation error, linear y-axis
        3. Percent conservation error, logarithmic y-axis
        4. Percent conservation error, linear y-axis

    The raw conservation-error data is shown only with scatter markers.
    The short and long moving averages remain as lines over the scatter data.

    Expected files in both the FOM and ROM error directories:

        error_cons.txt
        error_cons_percent.txt

    FOM data is loaded and validated, but no FOM conservation-error figures
    are added to the PDF.
    """
    fom_error_directory = Path(fom_error_directory)
    rom_error_directory = Path(rom_error_directory)

    if short_window <= 0 or long_window <= 0:
        raise ValueError(
            "Moving-average windows must be positive integers."
        )

    # ------------------------------------------------------------------
    # Load absolute conservation errors
    # ------------------------------------------------------------------

    fom_error_iterations, fom_error_values = load_error_signal_file(
        fom_error_directory / "error_cons.txt",
        delimiter=delimiter,
    )

    rom_error_iterations, rom_error_values = load_error_signal_file(
        rom_error_directory / "error_cons.txt",
        delimiter=delimiter,
    )

    # ------------------------------------------------------------------
    # Load percent conservation errors
    # ------------------------------------------------------------------

    fom_percent_iterations, fom_percent_values = load_error_signal_file(
        fom_error_directory / "error_cons_percent.txt",
        delimiter=delimiter,
    )

    rom_percent_iterations, rom_percent_values = load_error_signal_file(
        rom_error_directory / "error_cons_percent.txt",
        delimiter=delimiter,
    )

    # ------------------------------------------------------------------
    # Validate variable counts
    # ------------------------------------------------------------------

    if fom_error_values.shape[1] != rom_error_values.shaperaise ValueError(
            "FOM and ROM conservation-error files contain different "
            "numbers of conservative variables.\n"
            f"FOM absolute-error variables: {fom_error_values.shape[1]}\n"
            f"ROM absolute-error variables: {rom_error_values.shape[1]}"
        )

    if fom_percent_values.shape[1] != rom_percent_values.shaperaise ValueError(
            "FOM and ROM percent-conservation-error files contain different "
            "numbers of conservative variables.\n"
            f"FOM percent-error variables: {fom_percent_values.shape[1]}\n"
            f"ROM percent-error variables: {rom_percent_values.shape[1]}"
        )

    if rom_error_values.shape[1] != rom_percent_values.shaperaise ValueError(
            "ROM absolute and percent conservation-error files contain "
            "different numbers of conservative variables.\n"
            f"ROM absolute-error variables: {rom_error_values.shape[1]}\n"
            f"ROM percent-error variables: {rom_percent_values.shape[1]}"
        )

    if fom_error_values.shape[1] != fom_percent_values.shaperaise ValueError(
            "FOM absolute and percent conservation-error files contain "
            "different numbers of conservative variables.\n"
            f"FOM absolute-error variables: {fom_error_values.shape[1]}\n"
            f"FOM percent-error variables: {fom_percent_values.shape[1]}"
        )

    # Absolute values are used because conservation errors are magnitudes.
    # This also allows all positive, finite values to appear on log axes.
    fom_error_values = np.abs(fom_error_values)
    rom_error_values = np.abs(rom_error_values)
    fom_percent_values = np.abs(fom_percent_values)
    rom_percent_values = np.abs(rom_percent_values)

    default_names = [
        "Mass",
        "Momentum",
        "Energy",
        "Species",
    ]

    number_of_variables = rom_error_values.shape[1]

    field_names = (
        default_names
        if number_of_variables == len(default_names)
        else [
            f"Conservative field {index + 1}"
            for index in range(number_of_variables)
        ]
    )

    # Only scatter variants are generated. The moving averages are still
    # drawn as lines on top of the scatter points.
    axis_variants = (
        ("Logarithmic y-axis", True),
        ("Linear y-axis", False),
    )

    def add_rom_scatter_page(
        field_name,
        value_iterations,
        values,
        short_average,
        long_average,
        quantity_label,
        y_label,
        axis_label,
        use_log_y,
    ):
        """
        Add one ROM conservation-error scatter page.

        Raw values are shown as scatter points. Moving averages are shown
        as lines over the scatter points.
        """
        value_iterations = np.asarray(
            value_iterations,
            dtype=float,
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        short_average = np.asarray(
            short_average,
            dtype=float,
        )

        long_average = np.asarray(
            long_average,
            dtype=float,
        )

        if value_iterations.shape != values.shape:
            raise ValueError(
                f"{quantity_label} iteration and value shapes do not match "
                f"for {field_name}.\n"
                f"Iteration shape: {value_iterations.shape}\n"
                f"Value shape: {values.shape}"
            )

        if short_average.shape != values.shape:
            raise ValueError(
                f"Short moving-average shape does not match the "
                f"{quantity_label.lower()} values for {field_name}.\n"
                f"Moving-average shape: {short_average.shape}\n"
                f"Value shape: {values.shape}"
            )

        if long_average.shape != values.shape:
            raise ValueError(
                f"Long moving-average shape does not match the "
                f"{quantity_label.lower()} values for {field_name}.\n"
                f"Moving-average shape: {long_average.shape}\n"
                f"Value shape: {values.shape}"
            )

        if use_log_y:
            raw_plot_values = np.where(
                np.isfinite(values) & (values > 0.0),
                values,
                np.nan,
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

        fig, value_axis = plt.subplots(
            figsize=(12, 7)
        )

        raw_artist = value_axis.scatter(
            value_iterations,
            raw_plot_values,
            color="tab:red",
            s=10,
            alpha=0.50,
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

        value_axis.set_xlabel(
            "Iteration"
        )

        value_axis.set_ylabel(
            y_label
        )

        value_axis.set_title(
            f"ROM {quantity_label}: {field_name}\n"
            f"Scatter, {axis_label.lower()}"
        )

        if use_log_y:
            value_axis.set_yscale(
                "log"
            )

        value_axis.minorticks_on()

        value_axis.tick_params(
            axis="x",
            which="major",
            length=6,
        )

        value_axis.tick_params(
            axis="x",
            which="minor",
            length=3,
        )

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
            handles=[
                raw_artist,
                short_line,
                long_line,
            ],
            loc="best",
            fontsize=9,
        )

        fig.tight_layout()

        pdf.savefig(
            fig,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    print("============================================================")
    print("Appending ROM conservation-error scatter plots")
    print("FOM conservation-error plotting remains disabled")
    print(f"ROM variables: {number_of_variables}")
    print("Pages per ROM variable: 4")
    print("  1 absolute-error logarithmic scatter page")
    print("  1 absolute-error linear scatter page")
    print("  1 percent-error logarithmic scatter page")
    print("  1 percent-error linear scatter page")
    print(f"Moving-average windows: {short_window}, {long_window}")
    print("============================================================")

    # FOM conservation-error plots remain intentionally disabled.
    #
    # FOM absolute and percent data are still loaded so that:
    #
    #   1. Both required FOM files are checked.
    #   2. FOM and ROM conservative-variable counts are compared.
    #   3. Absolute and percent files are checked for consistent counts.
    #
    # FOM and ROM iteration counts are allowed to differ.

    for variable_index, field_name in enumerate(field_names):
        # --------------------------------------------------------------
        # Absolute conservation error
        # --------------------------------------------------------------

        absolute_values = rom_error_values[
            :,
            variable_index,
        ]

        absolute_short_average = compute_moving_average_ignore_nan(
            absolute_values,
            short_window,
        )

        absolute_long_average = compute_moving_average_ignore_nan(
            absolute_values,
            long_window,
        )

        for axis_label, use_log_y in axis_variants:
            add_rom_scatter_page(
                field_name=field_name,
                value_iterations=rom_error_iterations,
                values=absolute_values,
                short_average=absolute_short_average,
                long_average=absolute_long_average,
                quantity_label="Conservation Error History",
                y_label="Conservation error magnitude",
                axis_label=axis_label,
                use_log_y=use_log_y,
            )

            print(
                f"Added ROM absolute conservation-error page "
                f"{variable_index + 1}/{number_of_variables}: "
                f"{field_name}, scatter, {axis_label.lower()}"
            )

        # --------------------------------------------------------------
        # Percent conservation error
        # --------------------------------------------------------------

        percent_values = rom_percent_values[
            :,
            variable_index,
        ]

        percent_short_average = compute_moving_average_ignore_nan(
            percent_values,
            short_window,
        )

        percent_long_average = compute