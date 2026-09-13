#!/usr/bin/env python3
"""Generate FOM/ROM comparison and conservation-error plots.

All plot families are controlled by hard-coded True/False switches in the
USER CONFIGURATION section. Validation routines are grouped separately from
loading, numerical analysis, plotting, and orchestration routines.
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Plot-family switches. Set any entry to False to omit that page type.
PLOT_ENABLED = {
    "fom_rerun_history": False,
    "primitive_metrics": True,
    "conservative_integrals": False,
    "rom_conservation_error_log": False,
    "rom_conservation_error_linear": False,
    "baseline_comparison": False,
    "winsorized_mean_comparison": False,
    "winsorized_residual_metrics": False,
    "winsorized_standard_deviation": False,
    "winsorized_coefficient_of_variation": False,
    "adaptive_valley_band": False,
    "histogram_density_band": False,
    "variable_outlier_count": False,
    "rolling_rms": False,
    "rolling_mad_median": False,
    "rolling_normalized_mad_median": True,
    "rolling_median_standard_deviation": False,
    "rolling_median_coefficient_of_variation": False,
    "rolling_cv": True,
    "rolling_standard_deviation": False,
    "rolling_log_ratio": False,
    "interpolation_error_signals": False,
}

# Reported conservative values are already multiplied by volume.
CELL_VOLUME = 1.0

# Inclusive ranges. Use None or [] for one PDF containing all iterations.
PLOT_ITERATION_RANGES = []

PRIMITIVE_FIELD_NAMES = [
    "Density", "Velocity", "Pressure", "Temperature", "Reactant",
    "Heat release rate",
]
CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]

ROLLING_CV_WINDOWS = (1000,)
ROLLING_CV_MIN_MEAN_MAGNITUDE = 1.0e-30
ROLLING_CV_LOGY = False

ROLLING_MEDIAN_STAT_WINDOWS = (1000,)
ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE = 1.0e-30
ROLLING_MAD_LOGY = True
ROLLING_NORMALIZED_MAD_LOGY = False
ROLLING_MEDIAN_STD_LOGY = True
ROLLING_MEDIAN_CV_LOGY = True

ROLLING_STD_WINDOWS = (1000,)
ROLLING_STD_BASELINE_WINDOW = 5000
ROLLING_STD_LOGY = False

ROLLING_LOG_RATIO_WINDOWS = (1000,)
ROLLING_LOG_RATIO_BASELINE_WINDOW = 5000
ROLLING_LOG_RATIO_LOGY = False

BASELINE_COMPARISON_WINDOW = 1000
WINSORIZED_MEAN_WINDOW = 1000
WINSORIZED_TAIL_FRACTIONS = (0.05, 0.10, 0.20)
WINSORIZED_RESIDUAL_AVERAGE_WINDOW = 1000
WINSORIZED_RATIO_MIN_DENOMINATOR = 1.0e-30
WINSORIZED_STD_CV_WINDOW = 1000
WINSORIZED_CV_MIN_BASELINE_MAGNITUDE = 1.0e-30
WINSORIZED_STD_LOGY = True
WINSORIZED_CV_LOGY = True
ASYMMETRIC_MAD_LOWER_MULTIPLIER = 2.5
ASYMMETRIC_MAD_UPPER_MULTIPLIER = 2.5
ASYMMETRIC_MAD_NORMAL_SCALE = 1.4826

# Trailing sample count for the sliding-window density-histogram band.
HISTOGRAM_DENSITY_BAND_WINDOW = 1000

# Target expected sample count per SMOOTHED bin under Poisson sampling.
# Used to self-calibrate the box-filter smoothing radius: with this many
# expected samples, a truly occupied bin has under a 1% chance of a
# spuriously empty/low reading purely from sampling noise.
HISTOGRAM_DENSITY_BAND_TARGET_BIN_COUNT = 5.0

# Modified-z-score cutoff for flagging a curvature (change-in-decline-rate)
# value as an anomalous structural break rather than normal tail decline.
# 3.5 is the standard Iglewicz & Hoaglin convention, not a per-dataset
# tuned constant.
HISTOGRAM_DENSITY_BAND_CURVATURE_CUTOFF = 3.5

# Minimum number of accepted curvature samples before testing a candidate;
# a stability requirement so the local MAD estimate is not degenerate.
HISTOGRAM_DENSITY_BAND_MIN_CURVATURE_HISTORY = 3

ERROR_SIGNAL_GROUPS = (
    ("Primitive Variables - Maximum Error", "prim_interp_max.txt", "primitive"),
    ("Conservative Variables - Maximum Error", "cons_interp_max.txt", "conservative"),
    ("Primitive Variables - Average Error", "prim_interp_avg.txt", "primitive"),
    ("Conservative Variables - Average Error", "cons_interp_avg.txt", "conservative"),
)

PRIMITIVE_FILE_PATTERN = re.compile(r"^(\d+)iteration_prim\.npy$")
CONSERVATIVE_FILE_PATTERN = re.compile(r"^(\d+)iteration_cons\.npy$")


# =============================================================================
# VALIDATION AND CHECKING ROUTINES
# =============================================================================

def require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite; received {value!r}.")


def require_nonnegative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite; received {value!r}.")


def validate_directory(directory: Path, label: str) -> None:
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} directory does not exist:\n{directory}")


def validate_iteration_range(iteration_range, index: int = 0):
    if iteration_range is None:
        return None
    if len(iteration_range) != 2:
        raise ValueError(
            f"Iteration-range entry {index} must contain exactly two values: "
            f"{iteration_range!r}"
        )
    start, stop = iteration_range
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError(f"Iteration-range entry {index} must be finite: {iteration_range!r}")
    if start > stop:
        raise ValueError(f"Iteration-range start exceeds stop: {iteration_range!r}")
    return float(start), float(stop)


def validate_configuration() -> None:
    unknown = set(PLOT_ENABLED) - {
        "fom_rerun_history", "primitive_metrics", "conservative_integrals",
        "rom_conservation_error_log", "rom_conservation_error_linear",
        "baseline_comparison", "winsorized_mean_comparison",
        "winsorized_residual_metrics", "winsorized_standard_deviation",
        "winsorized_coefficient_of_variation", "adaptive_valley_band",
        "histogram_density_band", "variable_outlier_count",
        "rolling_rms",
        "rolling_mad_median", "rolling_normalized_mad_median",
        "rolling_median_standard_deviation",
        "rolling_median_coefficient_of_variation",
        "rolling_cv", "rolling_standard_deviation", "rolling_log_ratio",
        "interpolation_error_signals",
    }
    if unknown:
        raise ValueError(f"Unknown PLOT_ENABLED keys: {sorted(unknown)}")
    if not all(isinstance(value, bool) for value in PLOT_ENABLED.values()):
        raise TypeError("Every PLOT_ENABLED value must be exactly True or False.")
    require_positive(CELL_VOLUME, "CELL_VOLUME")
    for index, item in enumerate(configured_ranges()):
        validate_iteration_range(item, index)
    for window in (
        *ROLLING_CV_WINDOWS,
        *ROLLING_MEDIAN_STAT_WINDOWS,
        *ROLLING_STD_WINDOWS,
        *ROLLING_LOG_RATIO_WINDOWS,
    ):
        require_positive(window, "rolling window")
    require_nonnegative(
        ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE,
        "ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE",
    )
    for name, value in (
        ("ROLLING_MAD_LOGY", ROLLING_MAD_LOGY),
        ("ROLLING_NORMALIZED_MAD_LOGY", ROLLING_NORMALIZED_MAD_LOGY),
        ("ROLLING_MEDIAN_STD_LOGY", ROLLING_MEDIAN_STD_LOGY),
        ("ROLLING_MEDIAN_CV_LOGY", ROLLING_MEDIAN_CV_LOGY),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False.")
    require_positive(ROLLING_STD_BASELINE_WINDOW, "ROLLING_STD_BASELINE_WINDOW")
    require_positive(ROLLING_LOG_RATIO_BASELINE_WINDOW, "ROLLING_LOG_RATIO_BASELINE_WINDOW")
    require_positive(BASELINE_COMPARISON_WINDOW, "BASELINE_COMPARISON_WINDOW")
    require_positive(HISTOGRAM_DENSITY_BAND_WINDOW, "HISTOGRAM_DENSITY_BAND_WINDOW")
    require_positive(
        HISTOGRAM_DENSITY_BAND_TARGET_BIN_COUNT,
        "HISTOGRAM_DENSITY_BAND_TARGET_BIN_COUNT",
    )
    require_positive(
        HISTOGRAM_DENSITY_BAND_CURVATURE_CUTOFF,
        "HISTOGRAM_DENSITY_BAND_CURVATURE_CUTOFF",
    )
    require_positive(
        HISTOGRAM_DENSITY_BAND_MIN_CURVATURE_HISTORY,
        "HISTOGRAM_DENSITY_BAND_MIN_CURVATURE_HISTORY",
    )
    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")
    require_positive(
        WINSORIZED_RESIDUAL_AVERAGE_WINDOW,
        "WINSORIZED_RESIDUAL_AVERAGE_WINDOW",
    )
    require_nonnegative(
        WINSORIZED_RATIO_MIN_DENOMINATOR,
        "WINSORIZED_RATIO_MIN_DENOMINATOR",
    )
    require_positive(WINSORIZED_STD_CV_WINDOW, "WINSORIZED_STD_CV_WINDOW")
    require_nonnegative(
        WINSORIZED_CV_MIN_BASELINE_MAGNITUDE,
        "WINSORIZED_CV_MIN_BASELINE_MAGNITUDE",
    )
    if not isinstance(WINSORIZED_STD_LOGY, bool):
        raise TypeError("WINSORIZED_STD_LOGY must be True or False.")
    if not isinstance(WINSORIZED_CV_LOGY, bool):
        raise TypeError("WINSORIZED_CV_LOGY must be True or False.")
    if not WINSORIZED_TAIL_FRACTIONS:
        raise ValueError("WINSORIZED_TAIL_FRACTIONS must not be empty.")
    for fraction in WINSORIZED_TAIL_FRACTIONS:
        if not np.isfinite(fraction) or not 0.0 <= fraction < 0.5:
            raise ValueError(f"Invalid winsorized tail fraction: {fraction!r}")
    require_nonnegative(ROLLING_CV_MIN_MEAN_MAGNITUDE, "ROLLING_CV_MIN_MEAN_MAGNITUDE")


def validate_matching_3d_arrays(fom: np.ndarray, rom: np.ndarray, label: str) -> None:
    if fom.shape != rom.shape:
        raise ValueError(f"FOM and ROM {label} shapes differ: {fom.shape} versus {rom.shape}")
    if fom.ndim != 3:
        raise ValueError(
            f"Expected {label} shape (time, variable, cell); received {fom.shape}."
        )


def validate_history_shape(values, iterations, label: str) -> None:
    if np.asarray(values).shape != np.asarray(iterations).shape:
        raise ValueError(
            f"{label} shape {np.asarray(values).shape} does not match iteration "
            f"shape {np.asarray(iterations).shape}."
        )


def validate_windows(*named_windows) -> None:
    for name, value in named_windows:
        require_positive(value, name)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def configured_ranges():
    return [None] if not PLOT_ITERATION_RANGES else list(PLOT_ITERATION_RANGES)


def is_plot_enabled(name: str) -> bool:
    return PLOT_ENABLED[name]


def print_section(title: str, **items) -> None:
    print("=" * 68)
    print(title)
    for name, value in items.items():
        print(f"{name.replace('_', ' ').title()}: {value}")
    print("=" * 68, flush=True)


def iteration_mask(iterations, active_range):
    iterations = np.asarray(iterations)
    if active_range is None:
        return np.ones(iterations.shape, dtype=bool)
    start, stop = validate_iteration_range(active_range)
    return (iterations >= start) & (iterations <= stop)


def discover_iteration_files(directory: Path, pattern: re.Pattern):
    return {
        int(match.group(1)): path
        for path in directory.iterdir()
        if path.is_file() and (match := pattern.match(path.name))
    }


def output_filename(base_filename: Path, active_range):
    if active_range is None:
        return base_filename
    start, stop = active_range
    return base_filename.with_name(
        f"{base_filename.stem}_{start:g}_to_{stop:g}{base_filename.suffix}"
    )


def positive_or_nan(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def apply_standard_axes(axis, xlabel="Iteration") -> None:
    axis.set_xlabel(xlabel)
    axis.minorticks_on()
    axis.grid(which="major", axis="both", linestyle="-", alpha=0.35)
    axis.grid(which="minor", axis="x", linestyle=":", alpha=0.20)


def save_figure(pdf, fig, dpi=None) -> None:
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_common_npy_data(fom_dir, rom_dir, pattern, active_range, label):
    fom_dir, rom_dir = Path(fom_dir), Path(rom_dir)
    validate_directory(fom_dir, "FOM")
    validate_directory(rom_dir, "ROM")
    fom_files = discover_iteration_files(fom_dir, pattern)
    rom_files = discover_iteration_files(rom_dir, pattern)
    common = np.array(sorted(fom_files.keys() & rom_files.keys()), dtype=int)
    if common.size == 0:
        raise RuntimeError(f"No common FOM and ROM {label} files were found.")
    iterations = common[iteration_mask(common, active_range)]
    if iterations.size == 0:
        raise RuntimeError(f"No common {label} iterations fall in {active_range}.")
    fom = np.asarray([np.load(fom_files[int(i)]) for i in iterations])
    rom = np.asarray([np.load(rom_files[int(i)]) for i in iterations])
    validate_matching_3d_arrays(fom, rom, label)
    print_section(
        f"Loaded {label} data", common_iterations=len(iterations),
        first_iteration=iterations[0], last_iteration=iterations[-1], shape=fom.shape,
    )
    return fom, rom, iterations


def load_conservative_data_for_iterations(fom_dir, rom_dir, iterations):
    fom_files = discover_iteration_files(Path(fom_dir), CONSERVATIVE_FILE_PATTERN)
    rom_files = discover_iteration_files(Path(rom_dir), CONSERVATIVE_FILE_PATTERN)
    missing_fom = [int(i) for i in iterations if int(i) not in fom_files]
    missing_rom = [int(i) for i in iterations if int(i) not in rom_files]
    if missing_fom or missing_rom:
        parts = ["Conservative files are missing for selected primitive iterations."]
        if missing_fom:
            parts.append(f"Missing FOM iterations (first 20): {missing_fom[:20]}")
        if missing_rom:
            parts.append(f"Missing ROM iterations (first 20): {missing_rom[:20]}")
        raise FileNotFoundError("\n".join(parts))
    fom = np.asarray([np.load(fom_files[int(i)]) for i in iterations])
    rom = np.asarray([np.load(rom_files[int(i)]) for i in iterations])
    validate_matching_3d_arrays(fom, rom, "conservative data")
    return fom, rom


def load_error_signal_file(filename, active_range, delimiter=","):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Error-signal file does not exist:\n{filename}")
    data = np.loadtxt(filename, delimiter=delimiter, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected iteration plus signal columns in {filename}.")
    order = np.argsort(data[:, 0], kind="stable")
    iterations = np.asarray(data[order, 0], dtype=float)
    signals = np.asarray(data[order, 1:], dtype=float)
    if np.any(np.diff(iterations) == 0):
        raise ValueError(f"Duplicate iterations found in {filename}.")
    mask = iteration_mask(iterations, active_range)
    if not np.any(mask):
        raise ValueError(f"No iterations in {filename} fall in {active_range}.")
    return iterations[mask], signals[mask]


# =============================================================================
# NUMERICAL ANALYSIS
# =============================================================================

def compute_metric_histories(fom_data, rom_data, tolerance=1.0e-14):
    validate_matching_3d_arrays(fom_data, rom_data, "primitive data")
    n_time, n_variables, _ = fom_data.shape
    relative_l2 = np.full((n_variables, n_time), np.nan)
    correlation = np.full((n_variables, n_time), np.nan)
    for time_index in range(n_time):
        for variable_index in range(n_variables):
            fom = fom_data[time_index, variable_index]
            rom = rom_data[time_index, variable_index]
            valid = np.isfinite(fom) & np.isfinite(rom)
            if not np.any(valid):
                continue
            f, r = fom[valid], rom[valid]
            norm = np.linalg.norm(f)
            if norm > tolerance:
                relative_l2[variable_index, time_index] = np.linalg.norm(r - f) / norm
            if len(f) >= 2 and np.std(f) > tolerance and np.std(r) > tolerance:
                correlation[variable_index, time_index] = np.corrcoef(f, r)[0, 1]
    return relative_l2, correlation


def compute_conservative_integrals(fom, rom):
    validate_matching_3d_arrays(fom, rom, "conservative data")
    names = [
        CONSERVATIVE_FIELD_NAMES[i].lower() if i < len(CONSERVATIVE_FIELD_NAMES)
        else f"conservative_field_{i + 1}"
        for i in range(fom.shape[1])
    ]
    def integrate(data, label):
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{label} conservative data contain non-finite values.")
        return {
            name: CELL_VOLUME * np.sum(data[:, i, :], axis=1, dtype=np.float64)
            for i, name in enumerate(names)
        }
    return integrate(fom, "FOM"), integrate(rom, "ROM")


def moving_average(values, window):
    validate_windows(("moving-average window", window))
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    counts = np.r_[0, np.cumsum(finite.astype(int))]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - int(window))
    result = np.full(len(values), np.nan)
    valid = (counts[end] - counts[start]) > 0
    result[valid] = (sums[end] - sums[start])[valid] / (counts[end] - counts[start])[valid]
    return result


def moving_winsorized_means(values, window, tail_fractions):
    """Compute causal winsorized means and their rolling percentile bands.

    For tail fraction f, the lower and upper band edges are the rolling
    f and 1-f quantiles. Those same edges are used to clip the data before
    calculating the corresponding winsorized mean.
    """
    validate_windows(("winsorized-mean window", window))
    values = np.asarray(values, dtype=float)
    fractions = np.asarray(tuple(tail_fractions), dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("tail_fractions must be a nonempty 1-D sequence.")
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0) or np.any(fractions >= 0.5):
        raise ValueError("Each tail fraction must satisfy 0 <= fraction < 0.5.")

    means = np.full((fractions.size, len(values)), np.nan, dtype=float)
    lower_bands = np.full_like(means, np.nan)
    upper_bands = np.full_like(means, np.nan)
    requested_quantiles = np.concatenate((fractions, 1.0 - fractions))

    for index in range(1, len(values)):
        # Exclude the current sample so it cannot move its own mean or band.
        start = max(0, index - int(window))
        current = values[start:index]
        current = current[np.isfinite(current)]
        if current.size == 0:
            continue

        limits = np.quantile(current, requested_quantiles)
        lower_limits = limits[:fractions.size]
        upper_limits = limits[fractions.size:]
        lower_bands[:, index] = lower_limits
        upper_bands[:, index] = upper_limits

        for fraction_index, (lower, upper) in enumerate(
            zip(lower_limits, upper_limits)
        ):
            means[fraction_index, index] = np.mean(
                np.clip(current, lower, upper), dtype=np.float64
            )

    return means, lower_bands, upper_bands


def moving_average_selected(values, selected, window):
    """Compute a rolling mean using only samples selected by a Boolean mask."""
    values = np.asarray(values, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if values.shape != selected.shape:
        raise ValueError("Selected-value mask must match the value-array shape.")
    return moving_average(np.where(selected, values, np.nan), window)


def compute_winsorized_residual_metrics(
    values,
    winsorized_means,
    lower_bands,
    upper_bands,
    average_window,
    minimum_denominator,
):
    """Compute rolling tail and interior residual metrics.

    All residuals are nonnegative magnitudes, so upper and lower deviations
    cannot cancel one another.

    Mean-referenced metrics:
      tail:     abs(raw - winsorized mean), for points outside the band
      interior: abs(raw - winsorized mean), for points inside the band

    Limit-referenced metrics:
      tail: distance beyond the breached lower or upper band limit
      interior: distance to the nearest lower or upper band limit
    """
    validate_windows(("winsorized residual average window", average_window))
    require_nonnegative(minimum_denominator, "minimum residual-ratio denominator")

    values = np.asarray(values, dtype=float)
    means = np.asarray(winsorized_means, dtype=float)
    lower = np.asarray(lower_bands, dtype=float)
    upper = np.asarray(upper_bands, dtype=float)

    expected_shape = (means.shape[0], values.size)
    if means.ndim != 2 or lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError(
            "Winsorized means and bands must have matching shape "
            "(number_of_fractions, number_of_samples)."
        )

    number_of_fractions = means.shape[0]
    output_shape = (number_of_fractions, values.size)
    tail_mean_residual = np.full(output_shape, np.nan)
    interior_mean_residual = np.full(output_shape, np.nan)
    mean_residual_ratio = np.full(output_shape, np.nan)
    tail_limit_residual = np.full(output_shape, np.nan)
    interior_limit_residual = np.full(output_shape, np.nan)
    limit_residual_ratio = np.full(output_shape, np.nan)

    for fraction_index in range(number_of_fractions):
        baseline = means[fraction_index]
        lower_limit = lower[fraction_index]
        upper_limit = upper[fraction_index]

        valid = (
            np.isfinite(values)
            & np.isfinite(baseline)
            & np.isfinite(lower_limit)
            & np.isfinite(upper_limit)
            & (lower_limit <= upper_limit)
        )
        below = valid & (values < lower_limit)
        above = valid & (values > upper_limit)
        tail = below | above
        interior = valid & ~tail

        residual_from_mean = np.abs(values - baseline)
        tail_mean_residual[fraction_index] = moving_average_selected(
            residual_from_mean, tail, average_window
        )
        interior_mean_residual[fraction_index] = moving_average_selected(
            residual_from_mean, interior, average_window
        )

        valid_mean_ratio = (
            np.isfinite(tail_mean_residual[fraction_index])
            & np.isfinite(interior_mean_residual[fraction_index])
            & (interior_mean_residual[fraction_index] > minimum_denominator)
        )
        mean_residual_ratio[fraction_index, valid_mean_ratio] = (
            tail_mean_residual[fraction_index, valid_mean_ratio]
            / interior_mean_residual[fraction_index, valid_mean_ratio]
        )

        residual_from_limit = np.full(values.shape, np.nan)
        residual_from_limit[below] = lower_limit[below] - values[below]
        residual_from_limit[above] = values[above] - upper_limit[above]
        residual_from_limit[interior] = np.minimum(
            values[interior] - lower_limit[interior],
            upper_limit[interior] - values[interior],
        )

        tail_limit_residual[fraction_index] = moving_average_selected(
            residual_from_limit, tail, average_window
        )
        interior_limit_residual[fraction_index] = moving_average_selected(
            residual_from_limit, interior, average_window
        )

        valid_limit_ratio = (
            np.isfinite(tail_limit_residual[fraction_index])
            & np.isfinite(interior_limit_residual[fraction_index])
            & (interior_limit_residual[fraction_index] > minimum_denominator)
        )
        limit_residual_ratio[fraction_index, valid_limit_ratio] = (
            tail_limit_residual[fraction_index, valid_limit_ratio]
            / interior_limit_residual[fraction_index, valid_limit_ratio]
        )

    return {
        "tail_mean_residual": tail_mean_residual,
        "interior_mean_residual": interior_mean_residual,
        "mean_residual_ratio": mean_residual_ratio,
        "tail_limit_residual": tail_limit_residual,
        "interior_limit_residual": interior_limit_residual,
        "limit_residual_ratio": limit_residual_ratio,
    }


def rolling_winsorized_std_and_cv(
    values,
    winsorized_means,
    window,
    minimum_baseline_magnitude,
):
    """Compute rolling RMS spread about each causal winsorized mean and its CV.

    For fraction j and sample i, the spread is
        sqrt(mean((raw - winsorized_mean_j)**2))
    over the requested rolling window. All original raw values are retained,
    including tail values. The CV is that spread divided by the magnitude of
    the corresponding winsorized mean at sample i.
    """
    validate_windows(("winsorized std/CV window", window))
    require_nonnegative(
        minimum_baseline_magnitude,
        "minimum winsorized CV baseline magnitude",
    )

    values = np.asarray(values, dtype=float)
    means = np.asarray(winsorized_means, dtype=float)
    if means.ndim != 2 or means.shape[1] != values.size:
        raise ValueError(
            "winsorized_means must have shape "
            "(number_of_fractions, number_of_samples)."
        )

    spread = np.full(means.shape, np.nan, dtype=float)
    coefficient_of_variation = np.full(means.shape, np.nan, dtype=float)

    for fraction_index in range(means.shape[0]):
        baseline = means[fraction_index]
        residual = values - baseline
        spread[fraction_index] = np.sqrt(
            moving_average(residual ** 2, window)
        )

        valid = (
            np.isfinite(spread[fraction_index])
            & np.isfinite(baseline)
            & (np.abs(baseline) > minimum_baseline_magnitude)
        )
        coefficient_of_variation[fraction_index, valid] = (
            spread[fraction_index, valid] / np.abs(baseline[valid])
        )

    return spread, coefficient_of_variation


def exponential_moving_average(values, span):
    validate_windows(("EMA span", span))
    alpha = 2.0 / (float(span) + 1.0)
    result = np.full(len(values), np.nan)
    previous = np.nan
    for i, value in enumerate(np.asarray(values, dtype=float)):
        if np.isfinite(value):
            previous = value if not np.isfinite(previous) else alpha * value + (1-alpha) * previous
        result[i] = previous
    return result


def moving_median(values, window):
    validate_windows(("moving-median window", window))
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        current = values[max(0, i-window+1):i+1]
        current = current[np.isfinite(current)]
        if current.size:
            result[i] = np.median(current)
    return result


def moving_shorth(values, window):
    validate_windows(("moving-shorth window", window))
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        current = np.sort(values[max(0, i-window+1):i+1])
        current = current[np.isfinite(current)]
        n = current.size
        if not n:
            continue
        subset = (n + 1) // 2
        widths = current[subset-1:] - current[:n-subset+1]
        start = int(np.argmin(widths))
        result[i] = np.mean(current[start:start+subset])
    return result


def moving_asymmetric_mad_band(
    values,
    window,
    lower_multiplier=ASYMMETRIC_MAD_LOWER_MULTIPLIER,
    upper_multiplier=ASYMMETRIC_MAD_UPPER_MULTIPLIER,
):
    """Compute a causal asymmetric median/MAD band in log10 space.

    The lower and upper scales are estimated independently so the band can
    follow different amounts of spread below and above the moving median.
    Only finite positive samples are used because the calculation is done in
    log10 space.

    Returns values in the original signal units:
        median, lower_edge, upper_edge, band_mean, band_count
    """
    validate_windows(
        ("asymmetric-MAD window", window),
        ("asymmetric-MAD lower multiplier", lower_multiplier),
        ("asymmetric-MAD upper multiplier", upper_multiplier),
    )

    values = np.asarray(values, dtype=float)
    number_of_samples = len(values)
    median = np.full(number_of_samples, np.nan, dtype=float)
    lower_edge = np.full(number_of_samples, np.nan, dtype=float)
    upper_edge = np.full(number_of_samples, np.nan, dtype=float)
    band_mean = np.full(number_of_samples, np.nan, dtype=float)
    band_count = np.zeros(number_of_samples, dtype=np.int64)

    for index in range(number_of_samples):
        start = max(0, index - window + 1)
        current = values[start:index + 1]
        current = current[np.isfinite(current) & (current > 0.0)]
        if current.size == 0:
            continue

        log_values = np.log10(current)
        center_log = np.median(log_values)

        lower_deviations = center_log - log_values[log_values <= center_log]
        upper_deviations = log_values[log_values >= center_log] - center_log

        lower_mad = np.median(lower_deviations)
        upper_mad = np.median(upper_deviations)

        lower_scale = ASYMMETRIC_MAD_NORMAL_SCALE * lower_mad
        upper_scale = ASYMMETRIC_MAD_NORMAL_SCALE * upper_mad

        lower_log = center_log - lower_multiplier * lower_scale
        upper_log = center_log + upper_multiplier * upper_scale

        selected = (log_values >= lower_log) & (log_values <= upper_log)
        selected_values = current[selected]

        median[index] = 10.0 ** center_log
        lower_edge[index] = 10.0 ** lower_log
        upper_edge[index] = 10.0 ** upper_log
        band_count[index] = np.count_nonzero(selected)

        if selected_values.size:
            band_mean[index] = np.mean(selected_values, dtype=np.float64)

    return median, lower_edge, upper_edge, band_mean, band_count


def _freedman_diaconis_bin_edges(log_values):
    """Choose histogram bin edges from the data's own spread (no hardcoding).

    Bin width follows the Freedman-Diaconis rule (2*IQR / n^(1/3)), which
    self-calibrates to how spread out and how numerous the samples are. This
    keeps binning granularity a function of the data rather than a tuned
    per-dataset constant.
    """
    log_values = log_values[np.isfinite(log_values)]
    n = log_values.size
    if n < 2:
        raise ValueError("Need at least two finite values to build histogram bins.")

    lo, hi = float(log_values.min()), float(log_values.max())
    if hi <= lo:
        # All samples land on the same value; use a single degenerate bin.
        return np.array([lo, lo + 1.0])

    q75, q25 = np.percentile(log_values, [75, 25])
    iqr = q75 - q25
    if iqr > 0.0:
        bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    else:
        bin_width = 0.0

    if bin_width <= 0.0:
        # Degenerate IQR (e.g. more than half the samples share one value);
        # fall back to a data-driven bin count via Sturges' rule.
        n_bins = max(10, int(np.ceil(np.log2(n) + 1)))
    else:
        n_bins = max(10, int(np.ceil((hi - lo) / bin_width)))

    return np.linspace(lo, hi, n_bins + 1)


def _self_calibrated_smoothing_radius(window, n_bins, target_bin_sample_count):
    """Pick a box-filter radius (in bins) so a smoothed bin's *expected*
    sample count clears `target_bin_sample_count`, given how many samples
    the window holds and how many bins they are spread across. This makes
    the smoothing amount track the resolution automatically instead of
    being a fixed, dataset-tuned constant: finer bins (more of them) or
    smaller windows automatically call for more smoothing.
    """
    if n_bins <= 0:
        return 0
    average_count_per_bin = window / n_bins
    if average_count_per_bin <= 0.0:
        return 0
    kernel_width_bins = max(1, int(np.ceil(target_bin_sample_count / average_count_per_bin)))
    return (kernel_width_bins - 1) // 2


def _modified_zscore(history, candidate):
    """Return candidate's modified z-score (Iglewicz & Hoaglin) against a
    robust median/MAD summary of `history`. Returns 0 if there is no
    history yet, and +inf if the history is degenerate (zero MAD) and the
    candidate exceeds it -- any deviation from an exactly flat history is
    by definition anomalous.
    """
    if len(history) == 0:
        return 0.0
    history = np.asarray(history, dtype=float)
    center = np.median(history)
    mad = np.median(np.abs(history - center))
    if mad > 0.0:
        return 0.6745 * (candidate - center) / mad
    return 0.0 if candidate <= center else np.inf


def _walk_density_edge(smoothed_counts, start_index, step, cutoff, min_history):
    """Walk outward from a peak bin through smoothed bin counts, stopping
    at whichever comes first: a genuinely empty bin, or a statistically
    anomalous CHANGE in the rate of density decline (curvature).

    Testing curvature rather than the raw drop matters because a smooth,
    unimodal cluster's log-density declines at a smoothly accelerating
    rate moving into its tail (exactly quadratic for a Gaussian), so
    testing the raw drop against past drops eventually flags normal tail
    decline as anomalous and cuts the band too tight. The curvature
    (the change in the drop from one step to the next) stays roughly
    constant through a smooth, unimodal tail, and only spikes at a real
    structural break -- e.g. a shoulder or secondary population that
    never quite reaches zero density but is clearly not part of the same
    cluster.
    """
    index = start_index
    n_bins = len(smoothed_counts)
    drop_history = []
    curvature_history = []

    while True:
        next_index = index + step
        if next_index < 0 or next_index >= n_bins:
            break

        current = smoothed_counts[index]
        candidate = smoothed_counts[next_index]

        if candidate <= 0.0:
            break  # genuinely empty bin: unconditional stop

        if current > 0.0:
            drop = np.log(current) - np.log(candidate)
            if drop_history:
                curvature = drop - drop_history[-1]
                if (
                    len(curvature_history) >= min_history
                    and _modified_zscore(curvature_history, curvature) > cutoff
                ):
                    break
                curvature_history.append(curvature)
            drop_history.append(drop)

        index = next_index

    return index


def moving_histogram_density_band(
    values,
    window,
    target_bin_sample_count=HISTOGRAM_DENSITY_BAND_TARGET_BIN_COUNT,
    curvature_cutoff=HISTOGRAM_DENSITY_BAND_CURVATURE_CUTOFF,
    min_curvature_history=HISTOGRAM_DENSITY_BAND_MIN_CURVATURE_HISTORY,
):
    """Track the densest cluster's edges with an incrementally updated
    sliding-window histogram in log10 space.

    Bin edges are derived once (Freedman-Diaconis, see
    _freedman_diaconis_bin_edges) from the full finite-positive history, so
    no per-dataset threshold is hardcoded. At every sample, the trailing
    window's histogram is updated in O(1) (one bin increment for the
    entering sample, one bin decrement for the sample leaving the window),
    then box-filter smoothed (see _self_calibrated_smoothing_radius) before
    the peak and band edges are found.

    The band is the contiguous run of bins around the tallest (peak) bin
    reached by walking outward until either a bin is genuinely empty, or
    the CHANGE in the rate of density decline (curvature) becomes a
    statistical outlier relative to the walk's own history so far -- see
    _walk_density_edge. This catches structural breaks (shoulders,
    secondary populations) that never reach exactly zero density, not just
    literal gaps. Edges are still reported at the full, unsmoothed bin
    resolution.

    Cost: O(number_of_bins) per sample (histogram update, smoothing pass,
    and peak/edge search), versus O(window * log(window)) for resorting
    the trailing window from scratch at every step.

    Returns:
        lower_edge, upper_edge, band_count (each shape (len(values),))
    """
    validate_windows(("histogram-density-band window", window))
    require_positive(target_bin_sample_count, "target_bin_sample_count")
    require_positive(curvature_cutoff, "curvature_cutoff")
    require_positive(min_curvature_history, "min_curvature_history")

    values = np.asarray(values, dtype=float)
    n = len(values)

    lower_edge = np.full(n, np.nan, dtype=float)
    upper_edge = np.full(n, np.nan, dtype=float)
    band_count = np.zeros(n, dtype=np.int64)

    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size < 2:
        return lower_edge, upper_edge, band_count

    bin_edges = _freedman_diaconis_bin_edges(np.log10(finite_positive))
    n_bins = len(bin_edges) - 1

    smoothing_radius = _self_calibrated_smoothing_radius(
        window, n_bins, target_bin_sample_count
    )
    bin_positions = np.arange(n_bins)
    smoothing_lower = np.maximum(0, bin_positions - smoothing_radius)
    smoothing_upper = np.minimum(n_bins, bin_positions + smoothing_radius + 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    window_bins = deque()

    for index in range(n):
        value = values[index]
        entering_bin = None
        if np.isfinite(value) and value > 0.0:
            log_value = np.log10(value)
            entering_bin = int(
                np.clip(np.searchsorted(bin_edges, log_value, side="right") - 1,
                        0, n_bins - 1)
            )
            counts[entering_bin] += 1
        window_bins.append(entering_bin)

        if len(window_bins) > window:
            leaving_bin = window_bins.popleft()
            if leaving_bin is not None:
                counts[leaving_bin] -= 1

        if not np.any(counts):
            continue

        cumulative_counts = np.concatenate(([0], np.cumsum(counts)))
        smoothed_counts = (
            cumulative_counts[smoothing_upper] - cumulative_counts[smoothing_lower]
        ).astype(float)

        peak_bin = int(np.argmax(smoothed_counts))

        left_bin = _walk_density_edge(
            smoothed_counts, peak_bin, -1, curvature_cutoff, min_curvature_history
        )
        right_bin = _walk_density_edge(
            smoothed_counts, peak_bin, +1, curvature_cutoff, min_curvature_history
        )

        lower_edge[index] = 10.0 ** bin_edges[left_bin]
        upper_edge[index] = 10.0 ** bin_edges[right_bin + 1]
        band_count[index] = int(counts[left_bin:right_bin + 1].sum())

    return lower_edge, upper_edge, band_count


def rolling_statistic(values, window, statistic):
    validate_windows(("rolling window", window))
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(len(values)):
        current = values[max(0, i-window+1):i+1]
        current = current[np.isfinite(current)]
        if current.size:
            result[i] = statistic(current)
    return result


def rolling_rms_about_baseline(values, baseline, window):
    values, baseline = np.asarray(values, float), np.asarray(baseline, float)
    if values.shape != baseline.shape:
        raise ValueError("RMS values and baseline must have matching shapes.")
    return np.sqrt(moving_average((values-baseline)**2, window))


def rolling_cv(values, window, minimum_mean):
    require_nonnegative(minimum_mean, "minimum CV mean magnitude")
    def calculate(x):
        mean = np.mean(x)
        return np.std(x, ddof=0) / abs(mean) if abs(mean) >= minimum_mean else np.nan
    return rolling_statistic(values, window, calculate)


def rolling_median_spread_statistics(values, window, minimum_baseline_magnitude):
    """Return rolling median, MAD, normalized MAD, RMS spread, and CV.

    The RMS spread is the square root of the mean squared deviation from the
    window median. Normalized MAD and the coefficient of variation divide
    their spread measure by the magnitude of the same window median.
    """
    validate_windows(("rolling median-statistics window", window))
    require_nonnegative(
        minimum_baseline_magnitude,
        "minimum rolling-median baseline magnitude",
    )
    values = np.asarray(values, dtype=float)
    median = np.full(values.shape, np.nan, dtype=float)
    mad = np.full(values.shape, np.nan, dtype=float)
    normalized_mad = np.full(values.shape, np.nan, dtype=float)
    spread = np.full(values.shape, np.nan, dtype=float)
    coefficient_of_variation = np.full(values.shape, np.nan, dtype=float)

    for index in range(len(values)):
        start = max(0, index - int(window) + 1)
        current = values[start:index + 1]
        current = current[np.isfinite(current)]
        if current.size == 0:
            continue

        center = np.median(current)
        residual = current - center
        median[index] = center
        mad[index] = np.median(np.abs(residual))
        spread[index] = np.sqrt(np.mean(residual ** 2, dtype=np.float64))
        if abs(center) > minimum_baseline_magnitude:
            normalized_mad[index] = mad[index] / abs(center)
            coefficient_of_variation[index] = spread[index] / abs(center)

    return median, mad, normalized_mad, spread, coefficient_of_variation


def rolling_std_about_baseline(values, window, baseline_window):
    baseline = moving_average(values, baseline_window)
    return baseline, rolling_statistic(np.asarray(values)-baseline, window, lambda x: np.std(x))


def rolling_log_ratio(values, window, baseline_window):
    values = np.asarray(values, float)
    baseline = moving_median(values, baseline_window)
    ratio = np.full(len(values), np.nan)
    valid = np.isfinite(values) & np.isfinite(baseline) & (values > 0) & (baseline > 0)
    ratio[valid] = np.log10(values[valid] / baseline[valid])
    metric = rolling_statistic(ratio, window, lambda x: np.std(x))
    return baseline, ratio, metric


def interpolate_without_extrapolation(source_iterations, values, target_iterations):
    x, y = np.asarray(source_iterations, float), np.asarray(values, float)
    target = np.asarray(target_iterations, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.full(target.shape, np.nan)
    x, y = x[valid], y[valid]
    order = np.argsort(x, kind="stable")
    x, indices = np.unique(x[order], return_index=True)
    y = y[order][indices]
    if len(x) == 1:
        result = np.full(target.shape, np.nan)
        result[np.isclose(target, x[0])] = y[0]
        return result
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def rerun_mask(iterations, rerun_iterations):
    iterations = np.asarray(iterations, float)
    reruns = np.asarray(rerun_iterations, float)
    mask = np.zeros(iterations.shape, bool)
    valid = np.isfinite(iterations)
    valid_reruns = np.isfinite(reruns)
    if np.any(valid) and np.any(valid_reruns):
        mask[valid] = np.isin(np.rint(iterations[valid]).astype(np.int64),
                              np.unique(np.rint(reruns[valid_reruns]).astype(np.int64)))
    return mask


# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def plot_primitive_metrics(iterations, relative_l2, correlation, names, pdf):
    if len(names) != relative_l2.shape[0]:
        names = [f"Field {i+1}" for i in range(relative_l2.shape[0])]
    for i, name in enumerate(names):
        fig, error_axis = plt.subplots(figsize=(10, 6))
        corr_axis = error_axis.twinx()
        error_line, = error_axis.plot(iterations, 100*relative_l2[i], color="tab:red",
                                      linewidth=1.5, label=r"Relative $L_2$ error")
        corr_line, = corr_axis.plot(iterations, correlation[i], color="tab:blue",
                                    linewidth=1.5, label="Pearson correlation")
        error_axis.set_ylabel(r"Relative $L_2$ error [%]", color="tab:red")
        corr_axis.set_ylabel("Pearson correlation", color="tab:blue")
        error_axis.tick_params(axis="y", labelcolor="tab:red")
        corr_axis.tick_params(axis="y", labelcolor="tab:blue")
        corr_axis.set_ylim(1, 0)
        error_axis.set_title(name)
        apply_standard_axes(error_axis)
        error_axis.legend(handles=[error_line, corr_line], loc="best")
        save_figure(pdf, fig)


def plot_conservative_integrals(iterations, fom_integrals, rom_integrals, pdf, tolerance=1e-14):
    for key in fom_integrals:
        fom, rom = np.asarray(fom_integrals[key]), np.asarray(rom_integrals[key])
        validate_history_shape(fom, iterations, f"FOM {key}")
        validate_history_shape(rom, iterations, f"ROM {key}")
        error = np.full(fom.shape, np.nan)
        valid = np.isfinite(fom) & np.isfinite(rom) & (np.abs(fom) > tolerance)
        error[valid] = 100*np.abs(rom[valid]-fom[valid])/np.abs(fom[valid])
        fig, axis = plt.subplots(figsize=(10, 6)); error_axis = axis.twinx()
        fline, = axis.plot(iterations, fom, color="black", label="FOM")
        rline, = axis.plot(iterations, rom, color="tab:orange", ls="--", label="ROM")
        eline, = error_axis.plot(iterations, error, color="tab:red", ls=":", label="Relative error")
        axis.set_ylabel(f"Total {key}"); error_axis.set_ylabel("Relative error [%]", color="tab:red")
        axis.set_title(f"Domain-integrated {key}"); apply_standard_axes(axis)
        axis.legend(handles=[fline, rline, eline], loc="best")
        save_figure(pdf, fig)


def plot_winsorized_metric_family(
    iterations,
    metric,
    fractions,
    title,
    ylabel,
    pdf,
    logy=False,
):
    """Plot one rolling metric for every winsorized tail fraction."""
    metric = np.asarray(metric, dtype=float)
    fractions = tuple(fractions)
    if metric.shape != (len(fractions), len(iterations)):
        raise ValueError("Winsorized metric shape does not match iterations/fractions.")

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    line_styles = ("-", "--", "-.", ":")
    fig, axis = plt.subplots(figsize=(12, 7))

    for index, fraction in enumerate(fractions):
        plotted = positive_or_nan(metric[index]) if logy else metric[index]
        axis.plot(
            iterations,
            plotted,
            color=colors[index % len(colors)],
            linestyle=line_styles[index % len(line_styles)],
            linewidth=2.0,
            label=f"{100 * fraction:g}% tails",
        )

    if logy:
        finite_positive = metric[np.isfinite(metric) & (metric > 0.0)]
        if finite_positive.size:
            axis.set_yscale("log")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    apply_standard_axes(axis)
    axis.legend(loc="best")
    save_figure(pdf, fig, 150)


def plot_winsorized_residual_metric_pages(
    iterations,
    raw,
    winsorized_means,
    lower_bands,
    upper_bands,
    title,
    name,
    pdf,
):
    """Create four pages quantifying tail residual severity."""
    metrics = compute_winsorized_residual_metrics(
        raw,
        winsorized_means,
        lower_bands,
        upper_bands,
        WINSORIZED_RESIDUAL_AVERAGE_WINDOW,
        WINSORIZED_RATIO_MIN_DENOMINATOR,
    )
    window_text = f"rolling average window={WINSORIZED_RESIDUAL_AVERAGE_WINDOW}"

    plot_winsorized_metric_family(
        iterations,
        metrics["mean_residual_ratio"],
        WINSORIZED_TAIL_FRACTIONS,
        f"{title}: {name}\nTail/interior average absolute residual ratio about winsorized mean; {window_text}",
        "Tail mean residual / interior mean residual",
        pdf,
        logy=True,
    )
    plot_winsorized_metric_family(
        iterations,
        metrics["tail_mean_residual"],
        WINSORIZED_TAIL_FRACTIONS,
        f"{title}: {name}\nAverage absolute tail residual about winsorized mean; {window_text}",
        "Average tail residual [percentage points]",
        pdf,
        logy=True,
    )
    plot_winsorized_metric_family(
        iterations,
        metrics["limit_residual_ratio"],
        WINSORIZED_TAIL_FRACTIONS,
        f"{title}: {name}\nTail/interior average distance ratio relative to percentile limits; {window_text}",
        "Tail excess / interior distance to nearest limit",
        pdf,
        logy=True,
    )
    plot_winsorized_metric_family(
        iterations,
        metrics["tail_limit_residual"],
        WINSORIZED_TAIL_FRACTIONS,
        f"{title}: {name}\nAverage tail excess beyond percentile limit; {window_text}",
        "Average tail excess [percentage points]",
        pdf,
        logy=True,
    )


def plot_winsorized_std_cv_pages(
    iterations,
    raw,
    winsorized_means,
    title,
    name,
    pdf,
):
    """Plot spread and coefficient of variation for all tail fractions."""
    spread, coefficient_of_variation = rolling_winsorized_std_and_cv(
        raw,
        winsorized_means,
        WINSORIZED_STD_CV_WINDOW,
        WINSORIZED_CV_MIN_BASELINE_MAGNITUDE,
    )

    if is_plot_enabled("winsorized_standard_deviation"):
        plot_winsorized_metric_family(
            iterations,
            spread,
            WINSORIZED_TAIL_FRACTIONS,
            f"{title}: {name}\nRolling spread about winsorized means "
            f"(window={WINSORIZED_STD_CV_WINDOW})",
            "Rolling spread [percentage points]",
            pdf,
            logy=WINSORIZED_STD_LOGY,
        )

    if is_plot_enabled("winsorized_coefficient_of_variation"):
        plot_winsorized_metric_family(
            iterations,
            coefficient_of_variation,
            WINSORIZED_TAIL_FRACTIONS,
            f"{title}: {name}\nRolling coefficient of variation based on "
            f"winsorized means (window={WINSORIZED_STD_CV_WINDOW})",
            "Rolling CV = spread / |winsorized mean|",
            pdf,
            logy=WINSORIZED_CV_LOGY,
        )


def plot_baseline_pages(iterations, raw, title, name, pdf, baseline_window, rms_window):
    ma = moving_average(raw, baseline_window)
    ema = exponential_moving_average(raw, baseline_window)
    median = moving_median(raw, baseline_window)
    shorth = moving_shorth(raw, baseline_window)
    raw_log = positive_or_nan(raw)
    if is_plot_enabled("baseline_comparison"):
        fig, axis = plt.subplots(figsize=(12, 7))
        axis.scatter(iterations, raw_log, c="0.60", s=10, alpha=.4,
                     label="ROM percent conservation error", rasterized=True)
        axis.plot(iterations, positive_or_nan(ma), lw=2.4, label=f"Moving average ({baseline_window})")
        axis.plot(iterations, positive_or_nan(ema), lw=2.4, ls="--", label=f"EMA ({baseline_window})")
        axis.plot(iterations, positive_or_nan(median), lw=2.4, ls=":", label=f"Median ({baseline_window})")
        axis.plot(iterations, positive_or_nan(shorth), lw=2.4, ls="-.", label=f"Shorth ({baseline_window})")
        axis.set_yscale("log"); axis.set_ylabel("Conservation error [%]")
        axis.set_title(f"{title}: {name}\nCausal baseline comparison")
        apply_standard_axes(axis); axis.legend(loc="best"); save_figure(pdf, fig, 150)
    if is_plot_enabled("winsorized_mean_comparison"):
        winsorized_means, lower_bands, upper_bands = moving_winsorized_means(
            raw, WINSORIZED_MEAN_WINDOW, WINSORIZED_TAIL_FRACTIONS
        )
        fig, axis = plt.subplots(figsize=(12, 7))
        raw_artist = axis.scatter(
            iterations, raw_log, c="0.65", s=9, alpha=.22,
            label="Raw conservation error", rasterized=True, zorder=1,
        )
        handles = [raw_artist]
        colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
        line_styles = ("-", "--", "-.", ":")

        # Draw widest bands first so narrower bands remain visible on top.
        plot_order = np.argsort(WINSORIZED_TAIL_FRACTIONS)
        for index in plot_order:
            fraction = WINSORIZED_TAIL_FRACTIONS[index]
            color = colors[index % len(colors)]
            lower = positive_or_nan(lower_bands[index])
            upper = positive_or_nan(upper_bands[index])
            band = axis.fill_between(
                iterations,
                lower,
                upper,
                color=color,
                alpha=.07,
                linewidth=0.0,
                label=(f"{100*fraction:g}% tail band "
                       f"({100*fraction:g}th-{100*(1-fraction):g}th pct.)"),
                zorder=2 + index,
            )
            handles.append(band)
            axis.plot(
                iterations, lower, color=color, linewidth=.8,
                linestyle=line_styles[index % len(line_styles)], alpha=.70,
                zorder=6 + index,
            )
            axis.plot(
                iterations, upper, color=color, linewidth=.8,
                linestyle=line_styles[index % len(line_styles)], alpha=.70,
                zorder=6 + index,
            )

        for index, fraction in enumerate(WINSORIZED_TAIL_FRACTIONS):
            line, = axis.plot(
                iterations, positive_or_nan(winsorized_means[index]),
                color=colors[index % len(colors)],
                linestyle=line_styles[index % len(line_styles)], linewidth=2.2,
                label=f"Winsorized mean: {100*fraction:g}% tails",
                zorder=10 + index,
            )
            handles.append(line)

        axis.set_yscale("log")
        axis.set_ylabel("Conservation error [%]")
        axis.set_title(
            f"{title}: {name}\nCausal winsorized means and percentile bands "
            f"(window={WINSORIZED_MEAN_WINDOW})"
        )
        apply_standard_axes(axis)
        axis.legend(handles=handles, loc="best", fontsize=8, ncol=2)
        save_figure(pdf, fig, 150)

        if is_plot_enabled("winsorized_residual_metrics"):
            plot_winsorized_residual_metric_pages(
                iterations,
                raw,
                winsorized_means,
                lower_bands,
                upper_bands,
                title,
                name,
                pdf,
            )

        if (
            is_plot_enabled("winsorized_standard_deviation")
            or is_plot_enabled("winsorized_coefficient_of_variation")
        ):
            plot_winsorized_std_cv_pages(
                iterations,
                raw,
                winsorized_means,
                title,
                name,
                pdf,
            )
    if is_plot_enabled("adaptive_valley_band"):
        vm, left, right, mean, counts = moving_asymmetric_mad_band(raw, baseline_window)
        fig, axis = plt.subplots(figsize=(12, 7))
        axis.scatter(iterations, raw_log, c="0.65", s=9, alpha=.3, rasterized=True,
                     label="ROM percent conservation error")
        axis.fill_between(iterations, positive_or_nan(left), positive_or_nan(right),
                          color="tab:cyan", alpha=.2, label="Asymmetric median/MAD band")
        axis.plot(iterations, positive_or_nan(vm), color="tab:purple", ls=":", lw=2.4,
                  label="Moving log-space median")
        axis.plot(iterations, positive_or_nan(mean), color="black", lw=2.2,
                  label="Mean inside asymmetric MAD band")
        axis.set_yscale("log"); axis.set_ylabel("Conservation error [%]")
        axis.set_title(f"{title}: {name}\nAsymmetric median/MAD band in log10 space")
        apply_standard_axes(axis); axis.legend(loc="best"); save_figure(pdf, fig, 150)
        finite = counts[counts > 0]
        if finite.size:
            print(f"Asymmetric MAD band {name}: selected count median={np.median(finite):.1f}")
    if is_plot_enabled("rolling_rms"):
        median_rms = rolling_rms_about_baseline(
            raw,
            median,
            rms_window,
        )

        valid_median = (
            np.isfinite(median_rms)
            & np.isfinite(median)
            & (np.abs(median) > 1.0e-30)
        )

        median_rms[valid_median] /= np.abs(median[valid_median])
        median_rms[~valid_median] = np.nan        
        fig, axis = plt.subplots(figsize=(12, 7)); raw_axis = axis.twinx()
        rms_line, = axis.plot(
            iterations, median_rms, color="tab:purple", lw=2.4,
            label=f"RMS about moving median ({baseline_window})",
        )
        scat = raw_axis.scatter(iterations, raw_log, c="0.55", s=9, alpha=.3,
                                label="Raw conservation error", rasterized=True)
        raw_axis.set_yscale("log"); axis.set_ylabel("RMS [percentage points]")
        raw_axis.set_ylabel("Conservation error [%]")
        axis.set_title(
            f"{title}: {name}\nRolling RMS about moving-median baseline"
        )
        apply_standard_axes(axis); axis.legend(handles=[rms_line, scat], loc="best")
        save_figure(pdf, fig, 150)


def plot_histogram_density_band_page(
    iterations, raw, lower_edge, upper_edge, outside_band, title, name, pdf, window,
):
    """Plot raw data with the sliding-histogram density-band edges overlaid,
    highlighting points that fall outside the band.
    """
    raw_log = positive_or_nan(raw)
    lower_log = positive_or_nan(lower_edge)
    upper_log = positive_or_nan(upper_edge)
    inside = np.isfinite(raw_log) & ~outside_band
    outside = np.isfinite(raw_log) & outside_band

    fig, axis = plt.subplots(figsize=(12, 7))
    band = axis.fill_between(
        iterations, lower_log, upper_log,
        where=np.isfinite(lower_log) & np.isfinite(upper_log),
        color="tab:cyan", alpha=.20, label="Sliding-histogram density band",
    )
    inside_artist = axis.scatter(
        iterations[inside], raw_log[inside], c="0.55", s=9, alpha=.35,
        label="Inside band", rasterized=True,
    )
    handles = [band, inside_artist]
    if np.any(outside):
        outside_artist = axis.scatter(
            iterations[outside], raw_log[outside], c="tab:red", s=16, alpha=.85,
            label="Outside band", rasterized=True, zorder=5,
        )
        handles.append(outside_artist)
    axis.plot(iterations, lower_log, color="tab:cyan", lw=1.0, ls="--", alpha=.8)
    axis.plot(iterations, upper_log, color="tab:cyan", lw=1.0, ls="--", alpha=.8)
    axis.set_yscale("log")
    axis.set_ylabel("Conservation error [%]")
    axis.set_title(
        f"{title}: {name}\nSliding-histogram density band (window={window})"
    )
    apply_standard_axes(axis)
    axis.legend(handles=handles, loc="best")
    save_figure(pdf, fig, 150)


def plot_variable_outlier_count(iterations, outlier_masks, names, title, pdf):
    """Plot how many variables fall outside their own density band per
    iteration -- a single severity indicator across all tracked variables.
    """
    stacked = np.vstack(outlier_masks)
    count = stacked.sum(axis=0)
    fig, axis = plt.subplots(figsize=(12, 7))
    line, = axis.step(
        iterations, count, where="post", color="tab:red", lw=1.6,
        label="Variables outside density band",
    )
    axis.set_ylabel("Number of variables outside band")
    axis.set_ylim(-0.5, stacked.shape[0] + 0.5)
    axis.set_yticks(range(stacked.shape[0] + 1))
    axis.set_title(
        f"{title}\nOutlier severity: variables outside sliding-histogram band "
        f"({', '.join(names)})"
    )
    apply_standard_axes(axis)
    axis.legend(handles=[line], loc="best")
    save_figure(pdf, fig, 150)


def plot_single_rolling_metric(iterations, raw, baseline, metric, name, title,
                               metric_label, line_label, color, pdf, logy=False):
    fig, axis = plt.subplots(figsize=(12, 7)); metric_axis = axis.twinx()
    axis.set_zorder(1)
    metric_axis.set_zorder(2)
    metric_axis.patch.set_visible(False)
    raw_line, = axis.plot(
        iterations, raw, c="0.55", lw=1, alpha=.45,
        label="Raw error signal", zorder=1,
    )
    handles = [raw_line]
    if baseline is not None:
        baseline_line, = axis.plot(iterations, baseline, c="tab:green", lw=2.2, ls="-.",
                                   label="Local baseline")
        handles.append(baseline_line)
    metric_plot = positive_or_nan(metric) if logy else metric
    metric_line, = metric_axis.plot(
        iterations, metric_plot, color=color, lw=2.4,
        label=line_label, zorder=4,
    )
    handles.append(metric_line)
    finite = np.asarray(raw)[np.isfinite(raw)]
    if finite.size and np.all(finite > 0):
        axis.set_yscale("log")
    if logy:
        metric_axis.set_yscale("log")
    axis.set_ylabel("Raw error signal"); metric_axis.set_ylabel(metric_label, color=color)
    metric_axis.tick_params(axis="y", labelcolor=color)
    axis.set_title(f"{title}: {name}\n{line_label}")
    apply_standard_axes(axis); axis.legend(handles=handles, loc="best")
    save_figure(pdf, fig, 150)


def plot_rolling_metrics(iterations, raw, title, name, pdf):
    for window in ROLLING_MEDIAN_STAT_WINDOWS:
        median, mad, normalized_mad, spread, median_cv = (
            rolling_median_spread_statistics(
                raw, window, ROLLING_MEDIAN_STAT_MIN_BASELINE_MAGNITUDE
            )
        )
        if is_plot_enabled("rolling_mad_median"):
            plot_single_rolling_metric(
                iterations, raw, median, mad, name, title,
                "Rolling MAD [percentage points]",
                f"Rolling MAD about median ({window})",
                "tab:red", pdf, ROLLING_MAD_LOGY,
            )
        if is_plot_enabled("rolling_normalized_mad_median"):
            plot_single_rolling_metric(
                iterations, raw, median, normalized_mad, name, title,
                "Rolling normalized MAD = MAD / |median|",
                f"Rolling normalized MAD about median ({window})",
                "tab:brown", pdf, ROLLING_NORMALIZED_MAD_LOGY,
            )
        if is_plot_enabled("rolling_median_standard_deviation"):
            plot_single_rolling_metric(
                iterations, raw, median, spread, name, title,
                "Rolling spread [percentage points]",
                f"Rolling RMS spread about median ({window})",
                "tab:orange", pdf, ROLLING_MEDIAN_STD_LOGY,
            )
        if is_plot_enabled("rolling_median_coefficient_of_variation"):
            plot_single_rolling_metric(
                iterations, raw, median, median_cv, name, title,
                "Rolling CV = spread / |median|",
                f"Rolling median-based CV ({window})",
                "tab:purple", pdf, ROLLING_MEDIAN_CV_LOGY,
            )
    if is_plot_enabled("rolling_cv"):
        for window in ROLLING_CV_WINDOWS:
            metric = rolling_cv(raw, window, ROLLING_CV_MIN_MEAN_MAGNITUDE)
            plot_single_rolling_metric(iterations, raw, None, metric, name, title,
                "Rolling coefficient of variation", f"Rolling CV ({window})",
                "tab:purple", pdf, ROLLING_CV_LOGY)
    if is_plot_enabled("rolling_standard_deviation"):
        for window in ROLLING_STD_WINDOWS:
            baseline, metric = rolling_std_about_baseline(raw, window, ROLLING_STD_BASELINE_WINDOW)
            plot_single_rolling_metric(iterations, raw, baseline, metric, name, title,
                "Rolling std about baseline", f"Rolling residual std ({window})",
                "tab:orange", pdf, ROLLING_STD_LOGY)
    if is_plot_enabled("rolling_log_ratio"):
        for window in ROLLING_LOG_RATIO_WINDOWS:
            baseline, _, metric = rolling_log_ratio(raw, window, ROLLING_LOG_RATIO_BASELINE_WINDOW)
            plot_single_rolling_metric(iterations, raw, baseline, metric, name, title,
                "Rolling std [decades]", f"Rolling std log10(raw/median baseline) ({window})",
                "tab:blue", pdf, ROLLING_LOG_RATIO_LOGY)


def plot_rom_conservation_errors(rom_error_dir, pdf, active_range, reruns_by_variable,
                                 short_window, delimiter):
    rom_error_dir = Path(rom_error_dir)
    iterations, values = load_error_signal_file(
        rom_error_dir / "error_cons_perct.txt", active_range, delimiter)
    ma_iterations, saved_ma = load_error_signal_file(
        rom_error_dir / "error_cons_perct_ma.txt", active_range, delimiter)
    if values.shape != saved_ma.shape or not np.allclose(iterations, ma_iterations, atol=1e-9):
        raise ValueError("Conservation percent-error and saved-MA files are not aligned.")
    values, saved_ma = np.abs(values), np.abs(saved_ma)
    names = CONSERVATIVE_FIELD_NAMES if values.shape[1] == 4 else [
        f"Conservative field {i+1}" for i in range(values.shape[1])]
    variants = []
    if is_plot_enabled("rom_conservation_error_log"):
        variants.append(("Logarithmic y-axis", True))
    if is_plot_enabled("rom_conservation_error_linear"):
        variants.append(("Linear y-axis", False))
    outlier_masks = []
    for i, name in enumerate(names):
        raw, ma = values[:, i], saved_ma[:, i]
        reruns = reruns_by_variable[i] if i < len(reruns_by_variable) else []
        highlighted = rerun_mask(iterations, reruns)
        for label, use_log in variants:
            raw_plot = positive_or_nan(raw) if use_log else raw
            ma_plot = positive_or_nan(ma) if use_log else ma
            finite = np.isfinite(raw_plot)
            fig, axis = plt.subplots(figsize=(12, 7))
            regular_artist = axis.scatter(iterations[finite & ~highlighted], raw_plot[finite & ~highlighted],
                c="tab:red", s=10, alpha=.45, label="ROM percent conservation error", rasterized=True)
            handles = [regular_artist]
            if np.any(finite & highlighted):
                handles.append(axis.scatter(iterations[finite & highlighted], raw_plot[finite & highlighted],
                    c="black", s=38, label="FOM-rerun iteration", rasterized=True))
            ma_line, = axis.plot(iterations, ma_plot, c="darkblue", lw=3,
                                 label="Saved moving average")
            handles.append(ma_line)
            if use_log:
                axis.set_yscale("log")
            axis.set_ylabel("Conservation error [%]")
            axis.set_title(f"ROM Percent Conservation Error: {name}\n{label}")
            apply_standard_axes(axis); axis.legend(handles=handles, loc="best")
            save_figure(pdf, fig, 150)
        plot_baseline_pages(iterations, raw, "ROM Percent Conservation Error", name,
                            pdf, BASELINE_COMPARISON_WINDOW, short_window)
        plot_rolling_metrics(iterations, raw, "ROM Percent Conservation Error", name, pdf)

        if is_plot_enabled("histogram_density_band") or is_plot_enabled("variable_outlier_count"):
            lower_edge, upper_edge, _band_count = moving_histogram_density_band(
                raw, HISTOGRAM_DENSITY_BAND_WINDOW
            )
            band_valid = np.isfinite(raw) & np.isfinite(lower_edge) & np.isfinite(upper_edge)
            outside_band = band_valid & ((raw < lower_edge) | (raw > upper_edge))
            outlier_masks.append(outside_band)

            if is_plot_enabled("histogram_density_band"):
                plot_histogram_density_band_page(
                    iterations, raw, lower_edge, upper_edge, outside_band,
                    "ROM Percent Conservation Error", name, pdf,
                    HISTOGRAM_DENSITY_BAND_WINDOW,
                )

    if is_plot_enabled("variable_outlier_count") and outlier_masks:
        plot_variable_outlier_count(
            iterations, outlier_masks, names, "ROM Percent Conservation Error", pdf,
        )


def plot_interpolation_error_signals(metric_iterations, l2, correlation, names,
                                     error_dir, pdf, active_range, reruns,
                                     short_window, long_window, delimiter):
    for title, basename, kind in ERROR_SIGNAL_GROUPS:
        filename = Path(error_dir) / basename
        if not filename.is_file():
            print(f"Warning: missing error-signal file: {filename}")
            continue
        iterations, signals = load_error_signal_file(filename, active_range, delimiter)
        field_names = names if kind == "primitive" and len(names) == signals.shape[1] else [
            f"{kind.title()} field {i+1}" for i in range(signals.shape[1])]
        for i, field_name in enumerate(field_names):
            raw = signals[:, i]
            fig, axis = plt.subplots(figsize=(12, 7)); l2_axis = axis.twinx()
            corr_axis = axis.twinx(); corr_axis.spines["right"].set_position(("axes", 1.13))
            mask = rerun_mask(iterations, reruns); finite = np.isfinite(raw)
            handles = [axis.scatter(iterations[finite & ~mask], raw[finite & ~mask], c="0.55",
                                    s=10, alpha=.55, label="Raw error signal", rasterized=True)]
            if np.any(finite & mask):
                handles.append(axis.scatter(iterations[finite & mask], raw[finite & mask], c="black",
                                            s=38, label="FOM-rerun iteration", rasterized=True))
            handles += axis.plot(iterations, moving_average(raw, short_window), c="tab:orange",
                                 lw=2.8, label=f"Moving average ({short_window})")
            handles += axis.plot(iterations, moving_average(raw, long_window), c="tab:green",
                                 lw=3, ls="-.", label=f"Moving average ({long_window})")
            if i < l2.shape[0]:
                handles += l2_axis.plot(iterations, interpolate_without_extrapolation(
                    metric_iterations, 100*l2[i], iterations), c="tab:red", label=r"Relative $L_2$ error [%]")
                handles += corr_axis.plot(iterations, interpolate_without_extrapolation(
                    metric_iterations, correlation[i], iterations), c="tab:blue", label="Pearson correlation")
            axis.set_ylabel("Error signal"); l2_axis.set_ylabel(r"Relative $L_2$ error [%]")
            corr_axis.set_ylabel("Pearson correlation"); corr_axis.set_ylim(1, 0)
            axis.set_title(f"{title}: {field_name}"); apply_standard_axes(axis)
            axis.legend(handles, [h.get_label() for h in handles], loc="best", fontsize=9)
            fig.subplots_adjust(right=.80); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# FOM-RERUN COUNTER
# =============================================================================

def load_fom_rerun_counter(filename, active_range, delimiter=","):
    iterations, columns = load_error_signal_file(filename, active_range, delimiter)
    if columns.shape[1] != 1:
        raise ValueError("FOM-rerun counter must contain exactly one value column.")
    values = columns[:, 0]
    rounded = np.rint(values)
    if not np.all(np.isfinite(values)) or not np.allclose(values, rounded, atol=1e-12):
        raise ValueError("FOM-rerun counter values must be finite integers.")
    counter = rounded.astype(np.int64)
    if np.any(counter < 0) or np.any(np.diff(counter) < 0):
        raise ValueError("FOM-rerun counter must be nonnegative and nondecreasing.")
    changes = np.diff(counter, prepend=counter[0])
    rerun_iterations = iterations[changes > 0]
    pattern = re.compile(r"^\s*#\s*threshold_exceeded,iteration=([^,]+),variables=([^,]*),mask=([01](?:\|[01])*)\s*$")
    records = {}
    with Path(filename).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if not match:
                continue
            iteration = float(match.group(1)); key = int(np.rint(iteration))
            if not np.isclose(iteration, key, atol=1e-9):
                raise ValueError(f"Noninteger trigger iteration on line {line_number}.")
            mask = np.array([item == "1" for item in match.group(3).split("|")])
            old = records.get(key, np.zeros(len(mask), bool))
            size = max(len(old), len(mask)); old = np.pad(old, (0, size-len(old)))
            mask = np.pad(mask, (0, size-len(mask))); records[key] = old | mask
    n_variables = max(4, max((len(mask) for mask in records.values()), default=0))
    per_variable = [[] for _ in range(n_variables)]
    selected_min, selected_max = iterations[0], iterations[-1]
    for key, mask in sorted(records.items()):
        if selected_min <= key <= selected_max:
            for i, exceeded in enumerate(mask):
                if exceeded:
                    per_variable[i].append(float(key))
    return iterations, counter, rerun_iterations, tuple(np.asarray(v) for v in per_variable)


def reruns_per_window(iterations, counter, window):
    validate_windows(("FOM rerun window", window))
    x, counter = np.asarray(iterations, float), np.asarray(counter, np.int64)
    if x.ndim != 1 or x.shape != counter.shape or not len(x):
        raise ValueError("Counter arrays must be nonempty, 1-D, and matching.")
    increments = np.diff(counter, prepend=counter[0])
    prefix = np.r_[0, np.cumsum(increments)]
    result = np.full(x.shape, np.nan)
    for i, current in enumerate(x):
        if current-x[0] >= window:
            first = np.searchsorted(x, current-window, side="right")
            result[i] = prefix[i+1]-prefix[first]
    return result


def plot_fom_rerun_history(iterations, counter, pdf, window):
    recent = reruns_per_window(iterations, counter, window)
    fig, axis = plt.subplots(figsize=(12, 7)); recent_axis = axis.twinx()
    cumulative, = axis.step(iterations, counter, where="post", c="tab:blue", lw=2,
                            label="Cumulative FOM reruns")
    recent_line, = recent_axis.plot(iterations, recent, c="tab:purple", lw=2,
                                    label=f"Reruns in previous {window:g} iterations")
    axis.set_ylabel("Cumulative FOM reruns"); recent_axis.set_ylabel("Recent FOM reruns")
    axis.set_ylim(bottom=0); recent_axis.set_ylim(bottom=0)
    axis.set_title("Conservation-Error FOM Rerun History")
    apply_standard_axes(axis); axis.legend(handles=[cumulative, recent_line], loc="best")
    save_figure(pdf, fig)


# =============================================================================
# PROGRAM ORCHESTRATION
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate FOM/ROM metric and conservation-error PDFs."
    )
    parser.add_argument("fom_dir", type=Path)
    parser.add_argument("rom_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--error-short-window", type=int, default=1000)
    parser.add_argument("--error-long-window", type=int, default=5000)
    parser.add_argument("--error-delimiter", default=",")
    parser.add_argument("--fom-rerun-window", type=int, default=1000)
    return parser.parse_args()


def run_range(args, active_range):
    fom_dir = args.fom_dir.expanduser().resolve()
    rom_dir = args.rom_dir.expanduser().resolve()
    validate_directory(fom_dir, "FOM")
    validate_directory(rom_dir, "ROM")
    base_output = (args.output.expanduser().resolve() if args.output
                   else rom_dir / "FOM_ROM_metrics.pdf")
    pdf_filename = output_filename(base_output, active_range)
    pdf_filename.parent.mkdir(parents=True, exist_ok=True)
    print_section("Run configuration", fom_directory=fom_dir, rom_directory=rom_dir,
                  output_pdf=pdf_filename, iteration_range=active_range,
                  enabled_plots=[k for k, v in PLOT_ENABLED.items() if v])

    fom, rom, iterations = load_common_npy_data(
        fom_dir, rom_dir, PRIMITIVE_FILE_PATTERN, active_range, "primitive")
    relative_l2, correlation = compute_metric_histories(fom, rom)
    del fom, rom

    fom_integrals = rom_integrals = None
    if is_plot_enabled("conservative_integrals"):
        fom_cons, rom_cons = load_conservative_data_for_iterations(fom_dir, rom_dir, iterations)
        fom_integrals, rom_integrals = compute_conservative_integrals(fom_cons, rom_cons)
        del fom_cons, rom_cons

    rom_error_dir = rom_dir.parent / "error"
    counter_file = rom_error_dir / "error_cons_fom_iter_counter.txt"
    counter_iterations = counter_values = None
    rerun_iterations = np.array([])
    reruns_by_variable = tuple(np.array([]) for _ in range(4))
    if counter_file.is_file():
        try:
            counter_iterations, counter_values, rerun_iterations, reruns_by_variable = (
                load_fom_rerun_counter(counter_file, active_range, args.error_delimiter)
            )
        except (OSError, ValueError) as error:
            print(f"Warning: ignoring invalid counter file {counter_file}: {error}")

    with PdfPages(pdf_filename) as pdf:
        if is_plot_enabled("fom_rerun_history") and counter_iterations is not None:
            plot_fom_rerun_history(counter_iterations, counter_values, pdf,
                                   args.fom_rerun_window)
        if is_plot_enabled("primitive_metrics"):
            plot_primitive_metrics(iterations, relative_l2, correlation,
                                   PRIMITIVE_FIELD_NAMES, pdf)
        if is_plot_enabled("conservative_integrals"):
            plot_conservative_integrals(iterations, fom_integrals, rom_integrals, pdf)
        if any(is_plot_enabled(key) for key in (
            "rom_conservation_error_log", "rom_conservation_error_linear",
            "baseline_comparison", "winsorized_mean_comparison",
            "winsorized_residual_metrics", "winsorized_standard_deviation",
            "winsorized_coefficient_of_variation", "adaptive_valley_band",
            "histogram_density_band", "variable_outlier_count",
            "rolling_rms",
            "rolling_mad_median", "rolling_normalized_mad_median",
            "rolling_median_standard_deviation",
            "rolling_median_coefficient_of_variation",
            "rolling_cv", "rolling_standard_deviation", "rolling_log_ratio")):
            plot_rom_conservation_errors(rom_error_dir, pdf, active_range,
                reruns_by_variable, args.error_short_window, args.error_delimiter)
        if is_plot_enabled("interpolation_error_signals"):
            plot_interpolation_error_signals(iterations, relative_l2, correlation,
                PRIMITIVE_FIELD_NAMES, rom_error_dir, pdf, active_range,
                rerun_iterations, args.error_short_window, args.error_long_window,
                args.error_delimiter)
    print(f"Saved: {pdf_filename}", flush=True)


def main():
    args = parse_arguments()
    validate_configuration()
    validate_windows(
        ("--error-short-window", args.error_short_window),
        ("--error-long-window", args.error_long_window),
        ("--fom-rerun-window", args.fom_rerun_window),
    )
    for index, active_range in enumerate(configured_ranges()):
        active_range = validate_iteration_range(active_range, index)
        run_range(args, active_range)


if __name__ == "__main__":
    main()