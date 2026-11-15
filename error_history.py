#!/usr/bin/env python3
# Generate selected FOM/ROM comparison and conservation-error plots.

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# USER CONFIGURATION
# =============================================================================

PLOT_ENABLED = {
    "sampling_frequency": True,
    "fom_rerun_history": False,
    "primitive_metrics": True,
    "conservative_integrals": False,
    "rom_conservation_error_log": False,
    "rom_conservation_error_linear": False,
    "winsorized_mean_comparison": False,
    "histogram_density_band": False,
    "saved_short_mad_moving_average": True,
    "saved_first_mad_moving_average": True,
    "rolling_cv": False,
    "interpolation_error_signals": False,
    "conservation_error_by_phase": True,
    "conservation_error_phase_mad": True,
    "conservation_error_phase_quantiles": True,
    "conservation_error_phase_tail_body": False,
    "phase_mad_ratio_online": True,
    "phase_quarter_mad_ratio_online": True,
    "signed_conservation_error_cycle_sum": False,
}

# Reported conservative values are already multiplied by volume.
CELL_VOLUME = 1.0

# Inclusive ranges. Use None or [] for one PDF containing all iterations.
PLOT_ITERATION_RANGES = [(0, 100_000),
                            (0, 20_000),
                            (20_000, 60_000),
                            (80_000, 100_000)]

PRIMITIVE_FIELD_NAMES = [
    "Density",
    "Velocity",
    "Pressure",
    "Temperature",
    "Reactant",
    "Heat release rate",
]
CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]

ROLLING_CV_WINDOWS = (1000,)
ROLLING_CV_MIN_MEAN_MAGNITUDE = 1.0e-30
ROLLING_CV_LOGY = False

SAMPLING_FREQUENCY_FILENAME = "sampling_freq.txt"
SAMPLING_FREQUENCY_AVERAGE_WINDOW = 1000

# Window used when the short-MAD moving average was calculated during
# data collection. This value is used only for plot labels and does
# not cause the moving average to be recomputed during post-processing.
SHORT_MAD_MOVING_AVERAGE_WINDOW = 1000

# First iteration represented by the conservation-error history.
CONSERVATION_ERROR_START_ITERATION = 11

# Rolling sampling-cycle windows for the online last/first phase MAD ratio.
PHASE_MAD_RATIO_CYCLE_WINDOWS = (10, 25, 50, 100)


WINSORIZED_MEAN_WINDOW = 1000
WINSORIZED_TAIL_FRACTIONS = (0.05, 0.10, 0.20)

HISTOGRAM_DENSITY_BAND_WINDOW = 1000
HISTOGRAM_DENSITY_BAND_TARGET_BIN_COUNT = 5.0
HISTOGRAM_DENSITY_BAND_CURVATURE_CUTOFF = 3.5
HISTOGRAM_DENSITY_BAND_MIN_CURVATURE_HISTORY = 3

PRIMITIVE_FILE_PATTERN = re.compile(r"^(\d+)iteration_prim\.npy$")
CONSERVATIVE_FILE_PATTERN = re.compile(r"^(\d+)iteration_cons\.npy$")

ERROR_SIGNAL_GROUPS = (
    ("Primitive Variables - Maximum Error", "prim_interp_max.txt", "primitive"),
    ("Conservative Variables - Maximum Error", "cons_interp_max.txt", "conservative"),
    ("Primitive Variables - Average Error", "prim_interp_avg.txt", "primitive"),
    ("Conservative Variables - Average Error", "cons_interp_avg.txt", "conservative"),
)


# =============================================================================
# VALIDATION AND CHECKING ROUTINES
# =============================================================================

def require_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite; received {value!r}.")


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
        raise ValueError(
            f"Iteration-range entry {index} must be finite: {iteration_range!r}"
        )
    if start > stop:
        raise ValueError(f"Iteration-range start exceeds stop: {iteration_range!r}")
    return float(start), float(stop)


def validate_configuration() -> None:
    expected = {
        "sampling_frequency",
        "fom_rerun_history",
        "primitive_metrics",
        "conservative_integrals",
        "rom_conservation_error_log",
        "rom_conservation_error_linear",
        "winsorized_mean_comparison",
        "histogram_density_band",
        "saved_short_mad_moving_average",
        "saved_first_mad_moving_average",
        "rolling_cv",
        "interpolation_error_signals",
        "conservation_error_by_phase",
        "conservation_error_phase_mad",
        "conservation_error_phase_quantiles",
        "conservation_error_phase_tail_body",
        "phase_mad_ratio_online",
        "phase_quarter_mad_ratio_online",
        "signed_conservation_error_cycle_sum",
    }
    unknown = set(PLOT_ENABLED) - expected
    missing = expected - set(PLOT_ENABLED)
    if unknown:
        raise ValueError(f"Unknown PLOT_ENABLED keys: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing PLOT_ENABLED keys: {sorted(missing)}")
    if not all(isinstance(value, bool) for value in PLOT_ENABLED.values()):
        raise TypeError("Every PLOT_ENABLED value must be exactly True or False.")

    require_positive(CELL_VOLUME, "CELL_VOLUME")
    for index, item in enumerate(configured_ranges()):
        validate_iteration_range(item, index)

    for window in ROLLING_CV_WINDOWS:
        require_positive(window, "rolling window")

    if ROLLING_CV_MIN_MEAN_MAGNITUDE < 0:
        raise ValueError("ROLLING_CV_MIN_MEAN_MAGNITUDE must be nonnegative.")
    if not isinstance(ROLLING_CV_LOGY, bool):
        raise TypeError("ROLLING_CV_LOGY must be True or False.")

    require_positive(
        SAMPLING_FREQUENCY_AVERAGE_WINDOW,
        "SAMPLING_FREQUENCY_AVERAGE_WINDOW",
    )

    require_positive(
        SHORT_MAD_MOVING_AVERAGE_WINDOW,
        "SHORT_MAD_MOVING_AVERAGE_WINDOW",
    )
    if not isinstance(
        SHORT_MAD_MOVING_AVERAGE_WINDOW,
        int,
    ):
        raise TypeError(
            "SHORT_MAD_MOVING_AVERAGE_WINDOW must be an integer."
        )

    require_positive(WINSORIZED_MEAN_WINDOW, "WINSORIZED_MEAN_WINDOW")
    if not WINSORIZED_TAIL_FRACTIONS:
        raise ValueError("WINSORIZED_TAIL_FRACTIONS must not be empty.")
    for fraction in WINSORIZED_TAIL_FRACTIONS:
        if not np.isfinite(fraction) or not 0.0 <= fraction < 0.5:
            raise ValueError(f"Invalid winsorized tail fraction: {fraction!r}")

    require_positive(
        HISTOGRAM_DENSITY_BAND_WINDOW,
        "HISTOGRAM_DENSITY_BAND_WINDOW",
    )
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


def validate_matching_3d_arrays(
    fom: np.ndarray,
    rom: np.ndarray,
    label: str,
) -> None:
    if fom.shape != rom.shape:
        raise ValueError(
            f"FOM and ROM {label} shapes differ: {fom.shape} versus {rom.shape}"
        )
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
        f"Loaded {label} data",
        common_iterations=len(iterations),
        first_iteration=iterations[0],
        last_iteration=iterations[-1],
        shape=fom.shape,
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

            fom_valid = fom[valid]
            rom_valid = rom[valid]
            norm = np.linalg.norm(fom_valid)
            if norm > tolerance:
                relative_l2[variable_index, time_index] = (
                    np.linalg.norm(rom_valid - fom_valid) / norm
                )
            if (
                len(fom_valid) >= 2
                and np.std(fom_valid) > tolerance
                and np.std(rom_valid) > tolerance
            ):
                correlation[variable_index, time_index] = np.corrcoef(
                    fom_valid,
                    rom_valid,
                )[0, 1]

    return relative_l2, correlation


def moving_average(values, window):
    require_positive(window, "moving-average window")
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(finite, values, 0.0))]
    counts = np.r_[0, np.cumsum(finite.astype(int))]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - int(window))
    result = np.full(len(values), np.nan)
    valid = (counts[end] - counts[start]) > 0
    result[valid] = (
        (sums[end] - sums[start])[valid]
        / (counts[end] - counts[start])[valid]
    )
    return result


def interpolate_without_extrapolation(
    source_iterations,
    values,
    target_iterations,
):
    source = np.asarray(source_iterations, dtype=float)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target_iterations, dtype=float)
    valid = np.isfinite(source) & np.isfinite(values)
    if not np.any(valid):
        return np.full(target.shape, np.nan)

    source = source[valid]
    values = values[valid]
    order = np.argsort(source, kind="stable")
    source, unique_indices = np.unique(source[order], return_index=True)
    values = values[order][unique_indices]

    if len(source) == 1:
        result = np.full(target.shape, np.nan)
        result[np.isclose(target, source[0])] = values[0]
        return result

    return np.interp(target, source, values, left=np.nan, right=np.nan)


def rolling_statistic(values, window, statistic):
    require_positive(window, "rolling window")
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    window = int(window)

    for index in range(len(values)):
        current = values[max(0, index - window + 1):index + 1]
        current = current[np.isfinite(current)]
        if current.size:
            result[index] = statistic(current)
    return result


def rolling_cv(values, window, minimum_mean):
    if minimum_mean < 0:
        raise ValueError("minimum CV mean magnitude must be nonnegative.")

    def calculate(current):
        mean = np.mean(current)
        if abs(mean) >= minimum_mean:
            return np.std(current, ddof=0) / abs(mean)
        return np.nan

    return rolling_statistic(values, window, calculate)


def compute_conservative_integrals(fom, rom):
    validate_matching_3d_arrays(fom, rom, "conservative data")
    names = [
        CONSERVATIVE_FIELD_NAMES[i].lower()
        if i < len(CONSERVATIVE_FIELD_NAMES)
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


def moving_winsorized_means(values, window, tail_fractions):
    """Compute causal winsorized means and rolling percentile bands."""
    require_positive(window, "winsorized-mean window")
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


def _freedman_diaconis_bin_edges(log_values):
    log_values = log_values[np.isfinite(log_values)]
    n = log_values.size
    if n < 2:
        raise ValueError("Need at least two finite values to build histogram bins.")

    lo, hi = float(log_values.min()), float(log_values.max())
    if hi <= lo:
        return np.array([lo, lo + 1.0])

    q75, q25 = np.percentile(log_values, [75, 25])
    iqr = q75 - q25
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0)) if iqr > 0.0 else 0.0

    if bin_width <= 0.0:
        n_bins = max(10, int(np.ceil(np.log2(n) + 1)))
    else:
        n_bins = max(10, int(np.ceil((hi - lo) / bin_width)))

    return np.linspace(lo, hi, n_bins + 1)


def _self_calibrated_smoothing_radius(window, n_bins, target_bin_sample_count):
    if n_bins <= 0:
        return 0
    average_count_per_bin = window / n_bins
    if average_count_per_bin <= 0.0:
        return 0
    kernel_width_bins = max(
        1,
        int(np.ceil(target_bin_sample_count / average_count_per_bin)),
    )
    return (kernel_width_bins - 1) // 2


def _modified_zscore(history, candidate):
    if len(history) == 0:
        return 0.0
    history = np.asarray(history, dtype=float)
    center = np.median(history)
    mad = np.median(np.abs(history - center))
    if mad > 0.0:
        return 0.6745 * (candidate - center) / mad
    return 0.0 if candidate <= center else np.inf


def _walk_density_edge(smoothed_counts, start_index, step, cutoff, min_history):
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
            break

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
    """Track the densest cluster using a sliding histogram in log10 space."""
    require_positive(window, "histogram-density-band window")
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
        window,
        n_bins,
        target_bin_sample_count,
    )
    bin_positions = np.arange(n_bins)
    smoothing_lower = np.maximum(0, bin_positions - smoothing_radius)
    smoothing_upper = np.minimum(n_bins, bin_positions + smoothing_radius + 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    window_bins = deque()

    for index, value in enumerate(values):
        entering_bin = None
        if np.isfinite(value) and value > 0.0:
            log_value = np.log10(value)
            entering_bin = int(
                np.clip(
                    np.searchsorted(bin_edges, log_value, side="right") - 1,
                    0,
                    n_bins - 1,
                )
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
            cumulative_counts[smoothing_upper]
            - cumulative_counts[smoothing_lower]
        ).astype(float)

        peak_bin = int(np.argmax(smoothed_counts))
        left_bin = _walk_density_edge(
            smoothed_counts,
            peak_bin,
            -1,
            curvature_cutoff,
            min_curvature_history,
        )
        right_bin = _walk_density_edge(
            smoothed_counts,
            peak_bin,
            1,
            curvature_cutoff,
            min_curvature_history,
        )

        lower_edge[index] = 10.0 ** bin_edges[left_bin]
        upper_edge[index] = 10.0 ** bin_edges[right_bin + 1]
        band_count[index] = int(counts[left_bin:right_bin + 1].sum())

    return lower_edge, upper_edge, band_count


def rerun_mask(iterations, rerun_iterations):
    iterations = np.asarray(iterations, float)
    reruns = np.asarray(rerun_iterations, float)
    mask = np.zeros(iterations.shape, bool)
    valid = np.isfinite(iterations)
    valid_reruns = np.isfinite(reruns)
    if np.any(valid) and np.any(valid_reruns):
        mask[valid] = np.isin(
            np.rint(iterations[valid]).astype(np.int64),
            np.unique(np.rint(reruns[valid_reruns]).astype(np.int64)),
        )
    return mask


# =============================================================================
# PLOTTING ROUTINES
# =============================================================================

def plot_primitive_metrics(iterations, relative_l2, correlation, names, pdf):
    if len(names) != relative_l2.shape[0]:
        names = [f"Field {index + 1}" for index in range(relative_l2.shape[0])]

    for index, name in enumerate(names):
        fig, error_axis = plt.subplots(figsize=(10, 6))
        correlation_axis = error_axis.twinx()
        error_line, = error_axis.plot(
            iterations,
            100.0 * relative_l2[index],
            color="tab:red",
            linewidth=1.5,
            label=r"Relative $L_2$ error",
        )
        correlation_line, = correlation_axis.plot(
            iterations,
            correlation[index],
            color="tab:blue",
            linewidth=1.5,
            label="Pearson correlation",
        )
        error_axis.set_ylabel(r"Relative $L_2$ error [%]", color="tab:red")
        correlation_axis.set_ylabel("Pearson correlation", color="tab:blue")
        error_axis.tick_params(axis="y", labelcolor="tab:red")
        correlation_axis.tick_params(axis="y", labelcolor="tab:blue")
        correlation_axis.set_ylim(1, 0)
        error_axis.set_title(name)
        apply_standard_axes(error_axis)

        # Use the Pearson-correlation axis for horizontal gridlines.
        # Keep the vertical gridlines from the shared iteration axis.
        error_axis.grid(False, which="both", axis="y")
        correlation_axis.minorticks_on()
        correlation_axis.grid(
            which="major",
            axis="y",
            linestyle="-",
            alpha=0.35,
        )
        correlation_axis.grid(
            which="minor",
            axis="y",
            linestyle=":",
            alpha=0.20,
        )
        correlation_axis.set_axisbelow(True)

        error_axis.legend(
            handles=[error_line, correlation_line],
            loc="best",
        )
        save_figure(pdf, fig)


def plot_sampling_frequency(error_dir, pdf, active_range, delimiter):
    iterations, columns = load_error_signal_file(
        Path(error_dir) / SAMPLING_FREQUENCY_FILENAME,
        active_range,
        delimiter,
    )
    if columns.shape[1] != 1:
        raise ValueError(
            f"{SAMPLING_FREQUENCY_FILENAME} must contain exactly one "
            "sampling-frequency column."
        )
    frequency = columns[:, 0]
    if not np.all(np.isfinite(frequency)):
        raise ValueError("Sampling-frequency values must be finite.")
    if np.any(frequency <= 0):
        raise ValueError("Sampling-frequency values must be positive.")
    running_average = moving_average(
        frequency, SAMPLING_FREQUENCY_AVERAGE_WINDOW
    )
    fig, axis = plt.subplots(figsize=(12, 7))
    raw_artist = axis.scatter(
        iterations, frequency, color="0.55", s=10, alpha=0.45,
        label="Raw sampling frequency", rasterized=True,
    )
    average_line, = axis.plot(
        iterations, running_average, color="tab:blue", linewidth=2.5,
        label=f"Running average ({SAMPLING_FREQUENCY_AVERAGE_WINDOW})",
    )
    axis.set_ylabel("Sampling frequency [iterations]")
    axis.set_title("ROM Sampling Frequency")
    apply_standard_axes(axis)
    axis.legend(handles=[raw_artist, average_line], loc="best")
    save_figure(pdf, fig, 150)


def plot_single_rolling_metric(
    iterations,
    raw,
    baseline,
    metric,
    name,
    title,
    metric_label,
    line_label,
    color,
    pdf,
    logy=False,
):
    """Plot one rolling metric without a cumulative-sum curve."""
    fig, raw_axis = plt.subplots(figsize=(12, 7))
    metric_axis = raw_axis.twinx()

    raw_axis.set_zorder(1)
    metric_axis.set_zorder(2)
    metric_axis.patch.set_visible(False)

    raw_artist = raw_axis.scatter(
        iterations,
        raw,
        color="0.55",
        s=9,
        alpha=0.45,
        label=r"Raw absolute conservation error",
        rasterized=True,
        zorder=1,
    )
    handles = [raw_artist]

    if baseline is not None:
        baseline_line, = raw_axis.plot(
            iterations,
            baseline,
            color="tab:green",
            linewidth=2.2,
            linestyle="-.",
            label="Local baseline",
            zorder=2,
        )
        handles.append(baseline_line)

    metric_plot = positive_or_nan(metric) if logy else metric
    metric_line, = metric_axis.plot(
        iterations,
        metric_plot,
        color=color,
        linewidth=2.4,
        label=line_label,
        zorder=4,
    )
    handles.append(metric_line)

    finite_raw = np.asarray(raw)[np.isfinite(raw)]
    if finite_raw.size and np.all(finite_raw > 0):
        raw_axis.set_yscale("log")

    if logy:
        metric_axis.set_yscale("log")

    raw_axis.set_ylabel(
        r"Absolute conservation error"
    )
    metric_axis.set_ylabel(metric_label, color=color)
    metric_axis.tick_params(axis="y", labelcolor=color)

    raw_axis.set_title(f"{title}: {name}\n{line_label}")
    apply_standard_axes(raw_axis)
    raw_axis.legend(handles=handles, loc="best")

    fig.subplots_adjust(right=0.88)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_saved_mad_moving_average(
    iterations,
    raw,
    short_median,
    short_mad,
    short_mad_average,
    name,
    title,
    pdf,
):
    """
    Plot the saved short median, short normalized MAD, and saved MAD average.

    Left logarithmic axis:
        raw absolute conservation error
        saved short-window error median

    Right linear axis:
        saved short-window normalized MAD
        saved moving average of the short normalized MAD

    The moving-average values are read directly from the selected saved MAD file.
    No moving-average calculation is performed during post-processing.
    """
    iterations = np.asarray(
        iterations,
        dtype=float,
    )
    raw = np.asarray(
        raw,
        dtype=float,
    )
    short_median = np.asarray(
        short_median,
        dtype=float,
    )
    short_mad = np.asarray(
        short_mad,
        dtype=float,
    )
    short_mad_average = np.asarray(
        short_mad_average,
        dtype=float,
    )

    validate_history_shape(
        raw,
        iterations,
        f"{name} raw conservation error",
    )
    validate_history_shape(
        short_median,
        iterations,
        f"{name} short saved median",
    )
    validate_history_shape(
        short_mad,
        iterations,
        f"{name} short saved MAD",
    )
    validate_history_shape(
        short_mad_average,
        iterations,
        f"{name} saved short-MAD moving average",
    )

    fig, raw_axis = plt.subplots(
        figsize=(12, 7)
    )
    mad_axis = raw_axis.twinx()

    raw_axis.set_zorder(1)
    mad_axis.set_zorder(2)
    mad_axis.patch.set_visible(False)

    raw_artist = raw_axis.scatter(
        iterations,
        positive_or_nan(raw),
        color="0.65",
        s=9,
        alpha=0.30,
        label="Raw absolute conservation error",
        rasterized=True,
        zorder=1,
    )

    short_median_line, = raw_axis.plot(
        iterations,
        positive_or_nan(short_median),
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        alpha=0.95,
        label="Short-window error median (100)",
        zorder=2,
    )

    short_mad_line, = mad_axis.plot(
        iterations,
        short_mad,
        color="tab:brown",
        linewidth=1.4,
        alpha=0.55,
        label="Short-window normalized MAD (100)",
        zorder=4,
    )

    average_line, = mad_axis.plot(
        iterations,
        short_mad_average,
        color="tab:orange",
        linewidth=2.8,
        alpha=1.0,
        label=(
            "Saved moving average of short normalized MAD "
            f"({SHORT_MAD_MOVING_AVERAGE_WINDOW})"
        ),
        zorder=5,
    )

    raw_axis.set_yscale("log")
    raw_axis.set_ylabel(
        "Absolute conservation error"
    )

    mad_axis_color = "tab:orange"

    mad_axis.set_ylabel(
        "Short normalized MAD and saved moving average",
        color=mad_axis_color,
    )
    mad_axis.tick_params(
        axis="y",
        colors=mad_axis_color,
    )
    mad_axis.spines["right"].set_color(
        mad_axis_color
    )
    mad_axis.set_ylim(bottom=0.0)

    raw_axis.set_title(
        f"{title}: {name}\n"
        "Short normalized MAD and saved moving-average baseline"
    )

    apply_standard_axes(raw_axis)

    # Keep vertical iteration gridlines from the shared x-axis, but use the
    # right MAD axis for horizontal gridlines.
    raw_axis.grid(
        False,
        which="both",
        axis="y",
    )

    mad_axis.minorticks_on()
    mad_axis.grid(
        which="major",
        axis="y",
        linestyle="-",
        alpha=0.35,
    )
    mad_axis.grid(
        which="minor",
        axis="y",
        linestyle=":",
        alpha=0.20,
    )
    mad_axis.set_axisbelow(True)

    handles = [
        raw_artist,
        short_median_line,
        short_mad_line,
        average_line,
    ]

    raw_axis.legend(
        handles=handles,
        loc="best",
        fontsize=8,
    )

    finite_short_mad = short_mad[
        np.isfinite(short_mad)
    ]
    finite_average = short_mad_average[
        np.isfinite(short_mad_average)
    ]

    if (
        finite_short_mad.size
        and finite_average.size
    ):
        print_section(
            f"Saved short-MAD moving average: {name}",
            moving_average_window=(
                SHORT_MAD_MOVING_AVERAGE_WINDOW
            ),
            median_short_mad=float(
                np.median(finite_short_mad)
            ),
            maximum_short_mad=float(
                np.max(finite_short_mad)
            ),
            median_saved_short_mad_average=float(
                np.median(finite_average)
            ),
            maximum_saved_short_mad_average=float(
                np.max(finite_average)
            ),
        )

    fig.subplots_adjust(right=0.88)

    pdf.savefig(
        fig,
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)


def sampling_update_from_path(path):
    for part in reversed(Path(path).parts):
        match = re.search(r"(?:^|_)u(\d+)(?:_|$)", part)
        if match:
            sampling_update = int(match.group(1))
            if sampling_update <= 1:
                raise ValueError(
                    f"Sampling update in {part!r} must be greater than 1."
                )
            return sampling_update

    raise ValueError(
        f"Could not extract a sampling update from {path}; expected a path "
        "component such as s1_u5."
    )


def phase_error_data(iterations, raw, sampling_update, name):
    iterations = np.asarray(iterations, dtype=float)
    raw = np.asarray(raw, dtype=float)
    validate_history_shape(raw, iterations, f"{name} raw conservation error")

    rounded_iterations = np.rint(iterations)
    if not np.allclose(iterations, rounded_iterations, atol=1.0e-9):
        raise ValueError("Phase plotting requires integer iteration values.")

    phases = (
        rounded_iterations.astype(np.int64)
        - CONSERVATION_ERROR_START_ITERATION
        + 1
    ) % sampling_update
    valid = np.isfinite(raw) & (raw > 0.0) & (phases != 0)
    if not np.any(valid):
        raise ValueError(
            f"No positive finite {name} conservation-error values exist "
            "at nonzero phases."
        )

    phase_numbers = np.arange(1, sampling_update)
    phase_values = [raw[valid & (phases == phase)] for phase in phase_numbers]
    if any(values.size == 0 for values in phase_values):
        missing = [
            int(phase)
            for phase, values in zip(phase_numbers, phase_values)
            if values.size == 0
        ]
        raise ValueError(f"No {name} values were found for phases {missing}.")

    return phases, valid, phase_numbers, phase_values


def phase_ratio(values, label):
    values = np.asarray(values, dtype=float)
    baseline = values[0]
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(f"Phase-1 {label} must be positive and finite.")
    return values / baseline


def set_shared_ratio_scale(axis, *ratios):
    finite = np.concatenate([
        np.asarray(ratio, dtype=float)[np.isfinite(ratio)]
        for ratio in ratios
    ])
    maximum = max(1.0, float(np.max(finite)))
    axis.set_ylim(0.0, 1.05 * maximum)


def plot_phase_raw_baseline(axis, phases, valid, raw):
    return axis.scatter(
        phases[valid],
        raw[valid],
        color="tab:red",
        s=10,
        alpha=0.35,
        label="Raw conservation error",
        rasterized=True,
        zorder=1,
    )


def finish_phase_axes(axis, sampling_update):
    axis.set_yscale("log")
    axis.set_xticks(np.arange(1, sampling_update))
    axis.set_xlim(0.5, sampling_update - 0.5)
    axis.set_ylabel("Raw absolute conservation error")
    apply_standard_axes(axis, xlabel="Phase in sampling-update cycle")


def plot_conservation_error_by_phase(
    iterations,
    raw,
    sampling_update,
    name,
    pdf,
):
    phases, valid, phase_numbers, phase_values = phase_error_data(
        iterations,
        raw,
        sampling_update,
        name,
    )

    phase_medians = np.array([np.median(values) for values in phase_values])
    phase_means = np.array([np.mean(values) for values in phase_values])
    median_ratios = phase_ratio(phase_medians, "median")
    mean_ratios = phase_ratio(phase_means, "mean")

    fig, axis = plt.subplots(figsize=(12, 7))
    median_ratio_axis = axis.twinx()
    mean_ratio_axis = axis.twinx()
    mean_ratio_axis.spines["right"].set_position(("axes", 1.12))
    mean_ratio_axis.patch.set_visible(False)
    raw_artist = plot_phase_raw_baseline(axis, phases, valid, raw)
    median_line, = axis.plot(
        phase_numbers,
        phase_medians,
        color="tab:blue",
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Phase median",
        zorder=3,
    )
    mean_line, = axis.plot(
        phase_numbers,
        phase_means,
        color="tab:orange",
        marker="s",
        linewidth=2.5,
        markersize=6,
        label="Phase mean",
        zorder=3,
    )
    median_ratio_line, = median_ratio_axis.plot(
        phase_numbers,
        median_ratios,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label="Median / phase-1 median",
        zorder=4,
    )
    mean_ratio_line, = mean_ratio_axis.plot(
        phase_numbers,
        mean_ratios,
        color="tab:orange",
        linestyle="--",
        linewidth=1.8,
        label="Mean / phase-1 mean",
        zorder=4,
    )
    median_ratio_axis.axhline(
        1.0,
        color="tab:blue",
        linewidth=0.9,
        linestyle=":",
        alpha=0.75,
    )
    mean_ratio_axis.axhline(
        1.0,
        color="tab:orange",
        linewidth=0.9,
        linestyle=":",
        alpha=0.75,
    )
    median_ratio_axis.set_ylabel(
        "Median ratio to phase 1",
        color="tab:blue",
    )
    median_ratio_axis.tick_params(axis="y", colors="tab:blue")
    median_ratio_axis.spines["right"].set_color("tab:blue")
    mean_ratio_axis.set_ylabel(
        "Mean ratio to phase 1",
        color="tab:orange",
    )
    mean_ratio_axis.tick_params(axis="y", colors="tab:orange")
    mean_ratio_axis.spines["right"].set_color("tab:orange")
    set_shared_ratio_scale(median_ratio_axis, median_ratios)
    set_shared_ratio_scale(mean_ratio_axis, mean_ratios)
    finish_phase_axes(axis, sampling_update)
    axis.set_title(
        f"ROM Absolute Conservation Error by Sampling Phase: {name}\n"
        f"Sampling update u{sampling_update}; mean and median ratios"
    )
    axis.legend(
        handles=[
            raw_artist,
            median_line,
            mean_line,
            median_ratio_line,
            mean_ratio_line,
        ],
        loc="best",
        fontsize=8,
    )
    fig.subplots_adjust(right=0.80)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_conservation_error_phase_mad(
    iterations,
    raw,
    sampling_update,
    name,
    pdf,
):
    phases, valid, phase_numbers, phase_values = phase_error_data(
        iterations,
        raw,
        sampling_update,
        name,
    )
    phase_mads = np.array([
        np.median(np.abs(values - np.median(values)))
        for values in phase_values
    ])
    mad_ratios = phase_ratio(phase_mads, "MAD")

    fig, axis = plt.subplots(figsize=(12, 7))
    ratio_axis = axis.twinx()
    raw_artist = plot_phase_raw_baseline(axis, phases, valid, raw)
    mad_line, = axis.plot(
        phase_numbers,
        phase_mads,
        color="tab:purple",
        marker="D",
        linewidth=2.5,
        markersize=5,
        label="Phase MAD",
        zorder=3,
    )
    mad_ratio_line, = ratio_axis.plot(
        phase_numbers,
        mad_ratios,
        color="tab:purple",
        linestyle="--",
        linewidth=2.0,
        label="MAD / phase-1 MAD",
        zorder=4,
    )
    ratio_axis.axhline(1.0, color="0.25", linewidth=0.9, linestyle=":")
    ratio_axis.set_ylabel("MAD ratio to phase 1")
    set_shared_ratio_scale(ratio_axis, mad_ratios)
    finish_phase_axes(axis, sampling_update)
    axis.set_title(
        f"ROM Absolute Conservation Error by Sampling Phase: {name}\n"
        f"Sampling update u{sampling_update}; phase MAD"
    )
    axis.legend(
        handles=[raw_artist, mad_line, mad_ratio_line],
        loc="best",
        fontsize=8,
    )
    save_figure(pdf, fig, 150)


def plot_conservation_error_phase_tail_body(
    iterations,
    raw,
    sampling_update,
    name,
    pdf,
):
    phases, valid, phase_numbers, phase_values = phase_error_data(
        iterations,
        raw,
        sampling_update,
        name,
    )
    p95 = np.array([
        np.quantile(values, 0.95)
        for values in phase_values
    ])
    p80 = np.array([
        np.quantile(values, 0.80)
        for values in phase_values
    ])
    tail_to_body_ratio = p95 / p80

    fig, axis = plt.subplots(figsize=(12, 7))
    ratio_axis = axis.twinx()
    raw_artist = plot_phase_raw_baseline(axis, phases, valid, raw)
    p95_line, = axis.plot(
        phase_numbers,
        p95,
        color="tab:purple",
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="95th percentile",
        zorder=3,
    )
    p80_line, = axis.plot(
        phase_numbers,
        p80,
        color="tab:blue",
        marker="s",
        linewidth=2.2,
        markersize=5,
        label="80th percentile",
        zorder=3,
    )
    ratio_line, = ratio_axis.plot(
        phase_numbers,
        tail_to_body_ratio,
        color="black",
        marker="D",
        linestyle="--",
        linewidth=2.2,
        markersize=5,
        label="P95 / P80",
        zorder=4,
    )
    ratio_axis.axhline(
        1.0,
        color="0.25",
        linewidth=0.9,
        linestyle=":",
    )
    ratio_axis.set_ylabel("Tail-to-body ratio: P95 / P80")
    ratio_axis.set_ylim(
        0.0,
        1.05 * max(1.0, float(np.max(tail_to_body_ratio))),
    )
    finish_phase_axes(axis, sampling_update)
    axis.set_title(
        f"ROM Absolute Conservation Error by Sampling Phase: {name}\n"
        f"Sampling update u{sampling_update}; P95-to-P80 tail/body ratio"
    )
    axis.legend(
        handles=[raw_artist, p95_line, p80_line, ratio_line],
        loc="best",
        fontsize=8,
    )
    save_figure(pdf, fig, 150)


def plot_conservation_error_phase_quantiles(
    iterations,
    raw,
    sampling_update,
    name,
    pdf,
):
    phases, valid, phase_numbers, phase_values = phase_error_data(
        iterations,
        raw,
        sampling_update,
        name,
    )
    quantiles = (0.95, 0.90, 0.85, 0.80)
    colors = ("tab:purple", "tab:orange", "tab:green", "tab:blue")

    fig, axis = plt.subplots(figsize=(12, 7))
    raw_artist = plot_phase_raw_baseline(axis, phases, valid, raw)
    handles = [raw_artist]
    for quantile, color in zip(quantiles, colors):
        phase_quantiles = np.array([
            np.quantile(values, quantile)
            for values in phase_values
        ])
        line, = axis.plot(
            phase_numbers,
            phase_quantiles,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"{100 * quantile:g}th percentile",
            zorder=3,
        )
        handles.append(line)

    finish_phase_axes(axis, sampling_update)
    axis.set_title(
        f"ROM Absolute Conservation Error by Sampling Phase: {name}\n"
        f"Sampling update u{sampling_update}; upper quantiles"
    )
    axis.legend(handles=handles, loc="best", fontsize=8)
    save_figure(pdf, fig, 150)
def plot_online_phase_mad_ratio(
    iterations,
    raw,
    sampling_update,
    cycle_windows,
    name,
    pdf,
):
    iterations = np.asarray(iterations, dtype=float)
    raw = np.asarray(raw, dtype=float)
    validate_history_shape(raw, iterations, f"{name} raw conservation error")

    rounded_iterations = np.rint(iterations)
    if not np.allclose(iterations, rounded_iterations, atol=1.0e-9):
        raise ValueError("Online phase-MAD plotting requires integer iterations.")

    integer_iterations = rounded_iterations.astype(np.int64)
    phases = (
        integer_iterations
        - CONSERVATION_ERROR_START_ITERATION
        + 1
    ) % sampling_update
    first_phase = 1
    last_phase = sampling_update - 1

    for cycle_window in cycle_windows:
        require_positive(cycle_window, "online phase-MAD cycle window")
        cycle_window = int(cycle_window)
        first_values = deque(maxlen=cycle_window)
        last_values = deque(maxlen=cycle_window)
        ratio_iterations = []
        ratios = []
        first_mads = []
        last_mads = []

        for index in range(len(integer_iterations)):
            iteration = integer_iterations[index]
            value = raw[index]
            phase = phases[index]

            if (
                iteration < CONSERVATION_ERROR_START_ITERATION
                or not np.isfinite(value)
                or value <= 0.0
                or phase == 0
            ):
                continue

            if phase == first_phase:
                first_values.append(value)
            elif phase == last_phase:
                last_values.append(value)

                if (
                    len(first_values) < cycle_window
                    or len(last_values) < cycle_window
                ):
                    continue

                first_array = np.asarray(first_values, dtype=float)
                last_array = np.asarray(last_values, dtype=float)
                first_mad = np.median(
                    np.abs(first_array - np.median(first_array))
                )
                last_mad = np.median(
                    np.abs(last_array - np.median(last_array))
                )

                if not np.isfinite(first_mad) or first_mad <= 0.0:
                    continue

                ratio_iterations.append(iteration)
                ratios.append(last_mad / first_mad)
                first_mads.append(first_mad)
                last_mads.append(last_mad)

        if not ratios:
            print(
                f"Warning: no finite online phase-MAD ratios for {name} "
                f"with N={cycle_window}."
            )
            continue

        ratio_iterations = np.asarray(ratio_iterations, dtype=float)
        ratios = np.asarray(ratios, dtype=float)
        first_mads = np.asarray(first_mads, dtype=float)
        last_mads = np.asarray(last_mads, dtype=float)

        fig, mad_axis = plt.subplots(figsize=(12, 7))
        ratio_axis = mad_axis.twinx()
        first_line, = mad_axis.plot(
            ratio_iterations,
            positive_or_nan(first_mads),
            color="tab:blue",
            linewidth=1.8,
            label="Rolling phase-1 MAD",
        )
        last_line, = mad_axis.plot(
            ratio_iterations,
            positive_or_nan(last_mads),
            color="tab:orange",
            linewidth=1.8,
            label=f"Rolling phase-{last_phase} MAD",
        )
        ratio_line, = ratio_axis.plot(
            ratio_iterations,
            ratios,
            color="black",
            linewidth=2.0,
            label=f"Phase-{last_phase} MAD / phase-1 MAD",
        )
        ratio_axis.axhline(
            1.0,
            color="0.35",
            linewidth=0.9,
            linestyle=":",
        )
        mad_axis.set_yscale("log")
        mad_axis.set_ylabel("Rolling phase MAD")
        ratio_axis.set_ylabel("Last-phase MAD / first-phase MAD")
        ratio_axis.set_ylim(
            0.0,
            1.05 * max(1.0, float(np.max(ratios))),
        )
        mad_axis.set_title(
            f"Online Phase-MAD Ratio: {name}\n"
            f"Rolling window = {cycle_window} sampling cycles; "
            f"phase {last_phase} / phase 1"
        )
        apply_standard_axes(
            mad_axis,
            xlabel="Iteration at completion of last phase",
        )
        mad_axis.legend(
            handles=[first_line, last_line, ratio_line],
            loc="best",
            fontsize=8,
        )
        fig.subplots_adjust(right=0.88)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_online_quarter_phase_mad_ratio(
    iterations,
    raw,
    sampling_update,
    cycle_windows,
    name,
    pdf,
):
    iterations = np.asarray(iterations, dtype=float)
    raw = np.asarray(raw, dtype=float)
    validate_history_shape(raw, iterations, f"{name} raw conservation error")

    rounded_iterations = np.rint(iterations)
    if not np.allclose(iterations, rounded_iterations, atol=1.0e-9):
        raise ValueError("Online quarter-phase MAD plotting requires integer iterations.")

    integer_iterations = rounded_iterations.astype(np.int64)
    phases = (
        integer_iterations
        - CONSERVATION_ERROR_START_ITERATION
        + 1
    ) % sampling_update

    populated_phase_count = sampling_update - 1
    quarter_phase_count = max(1, int(np.ceil(0.25 * populated_phase_count)))
    first_phases = np.arange(1, quarter_phase_count + 1)
    last_phases = np.arange(
        sampling_update - quarter_phase_count,
        sampling_update,
    )
    last_phase = sampling_update - 1

    for cycle_window in cycle_windows:
        require_positive(cycle_window, "online quarter-phase MAD cycle window")
        cycle_window = int(cycle_window)
        first_cycles = deque(maxlen=cycle_window)
        last_cycles = deque(maxlen=cycle_window)
        current_first = []
        current_last = []
        ratio_iterations = []
        ratios = []
        first_mads = []
        last_mads = []

        for index in range(len(integer_iterations)):
            iteration = integer_iterations[index]
            value = raw[index]
            phase = phases[index]

            if iteration < CONSERVATION_ERROR_START_ITERATION:
                continue

            if phase == 1:
                current_first = []
                current_last = []

            if np.isfinite(value) and value > 0.0:
                if phase in first_phases:
                    current_first.append(value)
                if phase in last_phases:
                    current_last.append(value)

            if phase != last_phase:
                continue

            if (
                len(current_first) == len(first_phases)
                and len(current_last) == len(last_phases)
            ):
                first_cycles.append(tuple(current_first))
                last_cycles.append(tuple(current_last))

            if (
                len(first_cycles) < cycle_window
                or len(last_cycles) < cycle_window
            ):
                continue

            first_array = np.asarray(first_cycles, dtype=float).reshape(-1)
            last_array = np.asarray(last_cycles, dtype=float).reshape(-1)
            first_mad = np.median(
                np.abs(first_array - np.median(first_array))
            )
            last_mad = np.median(
                np.abs(last_array - np.median(last_array))
            )

            if not np.isfinite(first_mad) or first_mad <= 0.0:
                continue

            ratio_iterations.append(iteration)
            ratios.append(last_mad / first_mad)
            first_mads.append(first_mad)
            last_mads.append(last_mad)

        if not ratios:
            print(
                f"Warning: no finite online quarter-phase MAD ratios for "
                f"{name} with N={cycle_window}."
            )
            continue

        ratio_iterations = np.asarray(ratio_iterations, dtype=float)
        ratios = np.asarray(ratios, dtype=float)
        first_mads = np.asarray(first_mads, dtype=float)
        last_mads = np.asarray(last_mads, dtype=float)

        fig, mad_axis = plt.subplots(figsize=(12, 7))
        ratio_axis = mad_axis.twinx()
        first_line, = mad_axis.plot(
            ratio_iterations,
            positive_or_nan(first_mads),
            color="tab:blue",
            linewidth=1.8,
            label=(
                f"Rolling MAD, first 25% "
                f"(phases {first_phases[0]}-{first_phases[-1]})"
            ),
        )
        last_line, = mad_axis.plot(
            ratio_iterations,
            positive_or_nan(last_mads),
            color="tab:orange",
            linewidth=1.8,
            label=(
                f"Rolling MAD, last 25% "
                f"(phases {last_phases[0]}-{last_phases[-1]})"
            ),
        )
        ratio_line, = ratio_axis.plot(
            ratio_iterations,
            ratios,
            color="tab:green",
            linewidth=2.2,
            label="Last-25% MAD / first-25% MAD",
        )
        ratio_axis.axhline(
            1.0,
            color="tab:green",
            linewidth=0.9,
            linestyle=":",
            alpha=0.75,
        )
        mad_axis.set_yscale("log")
        mad_axis.set_ylabel("Rolling phase-group MAD")
        ratio_axis.set_ylabel(
            "Last-25% MAD / first-25% MAD",
            color="tab:green",
        )
        ratio_axis.tick_params(axis="y", colors="tab:green")
        ratio_axis.spines["right"].set_color("tab:green")
        ratio_axis.set_ylim(
            0.0,
            1.05 * max(1.0, float(np.max(ratios))),
        )
        mad_axis.set_title(
            f"Online First/Last Quarter Phase-MAD Ratio: {name}\n"
            f"Rolling window = {cycle_window} sampling cycles; "
            f"{quarter_phase_count} phases per quarter"
        )
        apply_standard_axes(
            mad_axis,
            xlabel="Iteration at completion of last phase",
        )
        mad_axis.legend(
            handles=[first_line, last_line, ratio_line],
            loc="best",
            fontsize=8,
        )
        fig.subplots_adjust(right=0.88)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)


def plot_signed_conservation_error_cycle_sum(
    iterations,
    signed_raw,
    sampling_update,
    name,
    pdf,
):
    iterations = np.asarray(iterations, dtype=float)
    signed_raw = np.asarray(signed_raw, dtype=float)
    validate_history_shape(
        signed_raw,
        iterations,
        f"{name} signed raw conservation error",
    )

    rounded_iterations = np.rint(iterations)
    if not np.allclose(iterations, rounded_iterations, atol=1.0e-9):
        raise ValueError("Cycle-sum plotting requires integer iteration values.")

    integer_iterations = rounded_iterations.astype(np.int64)
    phases = (
        integer_iterations
        - CONSERVATION_ERROR_START_ITERATION
        + 1
    ) % sampling_update
    valid = (
        np.isfinite(signed_raw)
        & (integer_iterations >= CONSERVATION_ERROR_START_ITERATION)
        & (phases != 0)
    )
    if not np.any(valid):
        raise ValueError(
            f"No finite signed {name} conservation-error values exist at or "
            f"after iteration {CONSERVATION_ERROR_START_ITERATION}."
        )

    cycle_numbers = (
        integer_iterations[valid] - CONSERVATION_ERROR_START_ITERATION
    ) // sampling_update
    unique_cycles = np.unique(cycle_numbers)
    cycle_starts = (
        CONSERVATION_ERROR_START_ITERATION
        + unique_cycles * sampling_update
    )
    cycle_sums = np.array([
        np.sum(signed_raw[valid][cycle_numbers == cycle], dtype=np.float64)
        for cycle in unique_cycles
    ])

    fig, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        cycle_starts,
        cycle_sums,
        color="tab:purple",
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Signed error sum per update cycle",
    )
    axis.set_yscale(
    "symlog",
    linthresh=1.0e-12,
    )
    axis.axhline(
        0.0,
        color="black",
        linewidth=0.9,
        alpha=0.65,
    )
    axis.set_ylabel("Sum of signed conservation error")
    axis.set_title(
        f"Signed Conservation-Error Sum per Update Cycle: {name}\n"
        f"Sampling update u{sampling_update}; first cycle starts at iteration "
        f"{CONSERVATION_ERROR_START_ITERATION}"
    )
    apply_standard_axes(axis, xlabel="First iteration in update cycle")
    axis.legend(loc="best")
    save_figure(pdf, fig, 150)


def plot_rolling_metrics(iterations, raw, title, name, pdf):
    if is_plot_enabled("rolling_cv"):
        for window in ROLLING_CV_WINDOWS:
            metric = rolling_cv(raw, window, ROLLING_CV_MIN_MEAN_MAGNITUDE)
            plot_single_rolling_metric(
                iterations,
                raw,
                None,
                metric,
                name,
                title,
                "Rolling coefficient of variation",
                f"Rolling CV ({window})",
                "tab:purple",
                pdf,
                ROLLING_CV_LOGY,
            )


def plot_conservative_integrals(
    iterations,
    fom_integrals,
    rom_integrals,
    pdf,
    tolerance=1.0e-14,
):
    for key in fom_integrals:
        fom = np.asarray(fom_integrals[key])
        rom = np.asarray(rom_integrals[key])
        validate_history_shape(fom, iterations, f"FOM {key}")
        validate_history_shape(rom, iterations, f"ROM {key}")

        error = np.full(fom.shape, np.nan)
        valid = np.isfinite(fom) & np.isfinite(rom) & (np.abs(fom) > tolerance)
        error[valid] = 100.0 * np.abs(rom[valid] - fom[valid]) / np.abs(fom[valid])

        fig, axis = plt.subplots(figsize=(10, 6))
        error_axis = axis.twinx()
        fline, = axis.plot(iterations, fom, color="black", label="FOM")
        rline, = axis.plot(
            iterations,
            rom,
            color="tab:orange",
            linestyle="--",
            label="ROM",
        )
        eline, = error_axis.plot(
            iterations,
            error,
            color="tab:red",
            linestyle=":",
            label="Relative error",
        )
        axis.set_ylabel(f"Total {key}")
        error_axis.set_ylabel("Relative error [%]", color="tab:red")
        axis.set_title(f"Domain-integrated {key}")
        apply_standard_axes(axis)
        axis.legend(handles=[fline, rline, eline], loc="best")
        save_figure(pdf, fig)


def plot_winsorized_mean_comparison(iterations, raw, title, name, pdf):
    winsorized_means, lower_bands, upper_bands = moving_winsorized_means(
        raw,
        WINSORIZED_MEAN_WINDOW,
        WINSORIZED_TAIL_FRACTIONS,
    )

    fig, axis = plt.subplots(figsize=(12, 7))
    raw_artist = axis.scatter(
        iterations,
        positive_or_nan(raw),
        color="0.65",
        s=9,
        alpha=0.22,
        label=r"Raw absolute conservation error",
        rasterized=True,
        zorder=1,
    )
    handles = [raw_artist]
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    line_styles = ("-", "--", "-.", ":")

    for index in np.argsort(WINSORIZED_TAIL_FRACTIONS):
        fraction = WINSORIZED_TAIL_FRACTIONS[index]
        color = colors[index % len(colors)]
        style = line_styles[index % len(line_styles)]
        lower = positive_or_nan(lower_bands[index])
        upper = positive_or_nan(upper_bands[index])

        band = axis.fill_between(
            iterations,
            lower,
            upper,
            color=color,
            alpha=0.07,
            linewidth=0.0,
            label=(
                f"{100 * fraction:g}% tail band "
                f"({100 * fraction:g}th-{100 * (1 - fraction):g}th pct.)"
            ),
            zorder=2 + index,
        )
        handles.append(band)
        axis.plot(
            iterations,
            lower,
            color=color,
            linewidth=0.8,
            linestyle=style,
            alpha=0.70,
            zorder=6 + index,
        )
        axis.plot(
            iterations,
            upper,
            color=color,
            linewidth=0.8,
            linestyle=style,
            alpha=0.70,
            zorder=6 + index,
        )

    for index, fraction in enumerate(WINSORIZED_TAIL_FRACTIONS):
        line, = axis.plot(
            iterations,
            positive_or_nan(winsorized_means[index]),
            color=colors[index % len(colors)],
            linestyle=line_styles[index % len(line_styles)],
            linewidth=2.2,
            label=f"Winsorized mean: {100 * fraction:g}% tails",
            zorder=10 + index,
        )
        handles.append(line)

    axis.set_yscale("log")
    axis.set_ylabel(
        r"Absolute conservation error"
    )
    axis.set_title(
        f"{title}: {name}\nCausal winsorized means and percentile bands "
        f"(window={WINSORIZED_MEAN_WINDOW})"
    )
    apply_standard_axes(axis)
    axis.legend(handles=handles, loc="best", fontsize=8, ncol=2)
    save_figure(pdf, fig, 150)


def plot_histogram_density_band_page(
    iterations,
    raw,
    lower_edge,
    upper_edge,
    outside_band,
    title,
    name,
    pdf,
    window,
):
    raw_log = positive_or_nan(raw)
    lower_log = positive_or_nan(lower_edge)
    upper_log = positive_or_nan(upper_edge)
    inside = np.isfinite(raw_log) & ~outside_band
    outside = np.isfinite(raw_log) & outside_band

    fig, axis = plt.subplots(figsize=(12, 7))
    band = axis.fill_between(
        iterations,
        lower_log,
        upper_log,
        where=np.isfinite(lower_log) & np.isfinite(upper_log),
        color="tab:cyan",
        alpha=0.20,
        label="Sliding-histogram density band",
    )
    inside_artist = axis.scatter(
        iterations[inside],
        raw_log[inside],
        color="0.55",
        s=9,
        alpha=0.35,
        label="Inside band",
        rasterized=True,
    )
    handles = [band, inside_artist]

    if np.any(outside):
        outside_artist = axis.scatter(
            iterations[outside],
            raw_log[outside],
            color="tab:red",
            s=16,
            alpha=0.85,
            label="Outside band",
            rasterized=True,
            zorder=5,
        )
        handles.append(outside_artist)

    axis.plot(
        iterations,
        lower_log,
        color="tab:cyan",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    axis.plot(
        iterations,
        upper_log,
        color="tab:cyan",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    axis.set_yscale("log")
    axis.set_ylabel(
        r"Absolute conservation error"
    )
    axis.set_title(
        f"{title}: {name}\nSliding-histogram density band (window={window})"
    )
    apply_standard_axes(axis)
    axis.legend(handles=handles, loc="best")
    save_figure(pdf, fig, 150)


def plot_rom_conservation_errors(
    rom_error_dir,
    pdf,
    active_range,
    reruns_by_variable,
    delimiter,
):
    rom_error_dir = Path(rom_error_dir)
    sampling_update = (
        sampling_update_from_path(rom_error_dir)
        if any(
            is_plot_enabled(key)
            for key in (
                "conservation_error_by_phase",
                "conservation_error_phase_mad",
                "conservation_error_phase_quantiles",
                "conservation_error_phase_tail_body",
                "phase_mad_ratio_online",
                "phase_quarter_mad_ratio_online",
                "signed_conservation_error_cycle_sum",
            )
        )
        else None
    )
    iterations, values = load_error_signal_file(
        rom_error_dir / "error_cons.txt",
        active_range,
        delimiter,
    )
    ma_iterations, saved_ma = load_error_signal_file(
        rom_error_dir / "error_cons_perct_ma.txt",
        active_range,
        delimiter,
    )

    if values.shape != saved_ma.shape or not np.allclose(
        iterations,
        ma_iterations,
        atol=1.0e-9,
    ):
        raise ValueError(
            "Raw absolute conservation-error and saved-MA files "
            "are not aligned."
        )

    def load_saved_mad_history(
        filename,
        includes_moving_average=False,
        require_raw_alignment=True,
    ):
        """
        Load a saved MAD history.

        Short-file layout when includes_moving_average is True:

            medians[0:N],
            normalized_mads[0:N],
            moving_averages[0:N]

        Long-file layout when includes_moving_average is False:

            medians[0:N],
            normalized_mads[0:N]
        """
        mad_iterations, mad_columns = load_error_signal_file(
            rom_error_dir / filename,
            active_range,
            delimiter,
        )

        number_of_variables = values.shape[1]
        number_of_groups = (
            3
            if includes_moving_average
            else 2
        )
        expected_columns = (
            number_of_groups
            * number_of_variables
        )

        if mad_columns.shape[1] != expected_columns:
            if includes_moving_average:
                expected_description = (
                    f"{number_of_variables} medians, followed by "
                    f"{number_of_variables} normalized MAD values, "
                    f"followed by {number_of_variables} saved "
                    "moving-average values"
                )
            else:
                expected_description = (
                    f"{number_of_variables} medians followed by "
                    f"{number_of_variables} normalized MAD values"
                )

            raise ValueError(
                f"Expected {filename} to contain "
                f"{expected_columns} value columns: "
                f"{expected_description}; received "
                f"{mad_columns.shape[1]}."
            )

        if require_raw_alignment and (
            iterations.shape != mad_iterations.shape
            or not np.allclose(
                iterations,
                mad_iterations,
                atol=1.0e-9,
            )
        ):
            raise ValueError(
                "Raw absolute conservation-error and "
                f"{filename} are not aligned."
            )

        median_stop = number_of_variables
        mad_stop = 2 * number_of_variables

        medians = mad_columns[
            :,
            :median_stop,
        ]
        normalized_mads = mad_columns[
            :,
            median_stop:mad_stop,
        ]

        moving_averages = None

        if includes_moving_average:
            moving_averages = mad_columns[
                :,
                mad_stop:3 * number_of_variables,
            ]

        return (
            mad_iterations,
            medians,
            normalized_mads,
            moving_averages,
        )

    saved_mad_iterations = None
    saved_median = None
    saved_normalized_mad = None
    saved_normalized_mad_average = None

    first_mad_iterations = None
    first_median = None
    first_normalized_mad = None
    first_normalized_mad_average = None

    if is_plot_enabled("saved_short_mad_moving_average"):
        (
            saved_mad_iterations,
            saved_median,
            saved_normalized_mad,
            saved_normalized_mad_average,
        ) = load_saved_mad_history(
            "error_cons_mad.txt",
            includes_moving_average=True,
            require_raw_alignment=True,
        )

    if is_plot_enabled("saved_first_mad_moving_average"):
        (
            first_mad_iterations,
            first_median,
            first_normalized_mad,
            first_normalized_mad_average,
        ) = load_saved_mad_history(
            "error_cons_first_mad.txt",
            includes_moving_average=True,
            require_raw_alignment=False,
        )

    signed_values = values.copy()
    values = np.abs(values)
    saved_ma = np.abs(saved_ma)
    names = (
        CONSERVATIVE_FIELD_NAMES
        if values.shape[1] == len(CONSERVATIVE_FIELD_NAMES)
        else [f"Conservative field {i + 1}" for i in range(values.shape[1])]
    )

    variants = []
    if is_plot_enabled("rom_conservation_error_log"):
        variants.append(("Logarithmic y-axis", True))
    if is_plot_enabled("rom_conservation_error_linear"):
        variants.append(("Linear y-axis", False))

    for i, name in enumerate(names):
        raw = values[:, i]
        signed_raw = signed_values[:, i]
        ma = saved_ma[:, i]
        reruns = reruns_by_variable[i] if i < len(reruns_by_variable) else []
        highlighted = rerun_mask(iterations, reruns)

        for label, use_log in variants:
            raw_plot = positive_or_nan(raw) if use_log else raw
            ma_plot = positive_or_nan(ma) if use_log else ma
            finite = np.isfinite(raw_plot)

            fig, axis = plt.subplots(figsize=(12, 7))
            regular_artist = axis.scatter(
                iterations[finite & ~highlighted],
                raw_plot[finite & ~highlighted],
                color="tab:red",
                s=10,
                alpha=0.45,
                label=r"ROM absolute conservation error",
                rasterized=True,
            )
            handles = [regular_artist]

            if np.any(finite & highlighted):
                handles.append(
                    axis.scatter(
                        iterations[finite & highlighted],
                        raw_plot[finite & highlighted],
                        color="black",
                        s=38,
                        label="FOM-rerun iteration",
                        rasterized=True,
                    )
                )

            ma_line, = axis.plot(
                iterations,
                ma_plot,
                color="darkblue",
                linewidth=3,
                label="Saved moving average",
            )
            handles.append(ma_line)

            if use_log:
                axis.set_yscale("log")
            axis.set_ylabel(
                r"Absolute conservation error"
            )
            axis.set_title(
                f"ROM Absolute Conservation Error: {name}\n{label}"
            )
            apply_standard_axes(axis)
            axis.legend(handles=handles, loc="best")
            save_figure(pdf, fig, 150)

            plot_combined_saved_mad_metric(
                iterations=saved_mad_iterations,
                raw=raw,
                short_median=saved_median[:, i],
                long_median=saved_median_long[:, i],
                short_mad=saved_normalized_mad[:, i],
                long_mad=saved_normalized_mad_long[:, i],
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
                logy=ROLLING_NORMALIZED_MAD_LOGY,
            )

            plot_saved_short_long_mad_ratio(
                iterations=saved_mad_iterations,
                raw=raw,
                short_median=saved_median[:, i],
                long_median=saved_median_long[:, i],
                short_mad=saved_normalized_mad[:, i],
                long_mad=saved_normalized_mad_long[:, i],
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
            )

        if is_plot_enabled("saved_short_mad_moving_average"):
            plot_saved_mad_moving_average(
                iterations=saved_mad_iterations,
                raw=raw,
                short_median=saved_median[:, i],
                short_mad=saved_normalized_mad[:, i],
                short_mad_average=saved_normalized_mad_average[:, i],
                name=name,
                title="ROM Absolute Conservation Error",
                pdf=pdf,
            )

        if is_plot_enabled("saved_first_mad_moving_average"):
            first_raw = interpolate_without_extrapolation(
                iterations,
                raw,
                first_mad_iterations,
            )
            plot_saved_mad_moving_average(
                iterations=first_mad_iterations,
                raw=first_raw,
                short_median=first_median[:, i],
                short_mad=first_normalized_mad[:, i],
                short_mad_average=first_normalized_mad_average[:, i],
                name=name,
                title="ROM First-After-Sampling Conservation Error",
                pdf=pdf,
            )

        if is_plot_enabled("conservation_error_by_phase"):
            plot_conservation_error_by_phase(
                iterations,
                raw,
                sampling_update,
                name,
                pdf,
            )

        if is_plot_enabled("conservation_error_phase_mad"):
            plot_conservation_error_phase_mad(
                iterations,
                raw,
                sampling_update,
                name,
                pdf,
            )

        if is_plot_enabled("conservation_error_phase_quantiles"):
            plot_conservation_error_phase_quantiles(
                iterations,
                raw,
                sampling_update,
                name,
                pdf,
            )

        if is_plot_enabled("conservation_error_phase_tail_body"):
            plot_conservation_error_phase_tail_body(
                iterations,
                raw,
                sampling_update,
                name,
                pdf,
            )

        if is_plot_enabled("phase_mad_ratio_online"):
            plot_online_phase_mad_ratio(
                iterations,
                raw,
                sampling_update,
                PHASE_MAD_RATIO_CYCLE_WINDOWS,
                name,
                pdf,
            )

        if is_plot_enabled("phase_quarter_mad_ratio_online"):
            plot_online_quarter_phase_mad_ratio(
                iterations,
                raw,
                sampling_update,
                PHASE_MAD_RATIO_CYCLE_WINDOWS,
                name,
                pdf,
            )

        if is_plot_enabled("signed_conservation_error_cycle_sum"):
            plot_signed_conservation_error_cycle_sum(
                iterations,
                signed_raw,
                sampling_update,
                name,
                pdf,
            )

        plot_rolling_metrics(
            iterations,
            raw,
            "ROM Absolute Conservation Error",
            name,
            pdf,
        )

        if is_plot_enabled("winsorized_mean_comparison"):
            plot_winsorized_mean_comparison(
                iterations,
                raw,
                "ROM Absolute Conservation Error",
                name,
                pdf,
            )

        if is_plot_enabled("histogram_density_band"):
            lower_edge, upper_edge, _ = moving_histogram_density_band(
                raw,
                HISTOGRAM_DENSITY_BAND_WINDOW,
            )
            band_valid = (
                np.isfinite(raw)
                & np.isfinite(lower_edge)
                & np.isfinite(upper_edge)
            )
            outside_band = band_valid & (
                (raw < lower_edge) | (raw > upper_edge)
            )
            plot_histogram_density_band_page(
                iterations,
                raw,
                lower_edge,
                upper_edge,
                outside_band,
                "ROM Absolute Conservation Error",
                name,
                pdf,
                HISTOGRAM_DENSITY_BAND_WINDOW,
            )


def plot_interpolation_error_signals(
    metric_iterations,
    relative_l2,
    correlation,
    names,
    error_dir,
    pdf,
    active_range,
    reruns,
    short_window,
    long_window,
    delimiter,
):
    for title, basename, kind in ERROR_SIGNAL_GROUPS:
        filename = Path(error_dir) / basename
        if not filename.is_file():
            print(f"Warning: missing error-signal file: {filename}")
            continue

        iterations, signals = load_error_signal_file(
            filename,
            active_range,
            delimiter,
        )
        field_names = (
            names
            if kind == "primitive" and len(names) == signals.shape[1]
            else [
                f"{kind.title()} field {index + 1}"
                for index in range(signals.shape[1])
            ]
        )

        for index, field_name in enumerate(field_names):
            raw = signals[:, index]
            fig, signal_axis = plt.subplots(figsize=(12, 7))
            l2_axis = signal_axis.twinx()
            correlation_axis = signal_axis.twinx()
            correlation_axis.spines["right"].set_position(("axes", 1.13))

            highlighted = rerun_mask(iterations, reruns)
            finite = np.isfinite(raw)
            handles = [
                signal_axis.scatter(
                    iterations[finite & ~highlighted],
                    raw[finite & ~highlighted],
                    color="0.55",
                    s=10,
                    alpha=0.55,
                    label="Raw error signal",
                    rasterized=True,
                )
            ]

            if np.any(finite & highlighted):
                handles.append(
                    signal_axis.scatter(
                        iterations[finite & highlighted],
                        raw[finite & highlighted],
                        color="black",
                        s=38,
                        label="FOM-rerun iteration",
                        rasterized=True,
                    )
                )

            handles.extend(
                signal_axis.plot(
                    iterations,
                    moving_average(raw, short_window),
                    color="tab:orange",
                    linewidth=2.8,
                    label=f"Moving average ({short_window})",
                )
            )
            handles.extend(
                signal_axis.plot(
                    iterations,
                    moving_average(raw, long_window),
                    color="tab:green",
                    linewidth=3,
                    linestyle="-.",
                    label=f"Moving average ({long_window})",
                )
            )

            if index < relative_l2.shape[0]:
                handles.extend(
                    l2_axis.plot(
                        iterations,
                        interpolate_without_extrapolation(
                            metric_iterations,
                            100.0 * relative_l2[index],
                            iterations,
                        ),
                        color="tab:red",
                        label=r"Relative $L_2$ error [%]",
                    )
                )
                handles.extend(
                    correlation_axis.plot(
                        iterations,
                        interpolate_without_extrapolation(
                            metric_iterations,
                            correlation[index],
                            iterations,
                        ),
                        color="tab:blue",
                        label="Pearson correlation",
                    )
                )

            signal_axis.set_ylabel("Error signal")
            l2_axis.set_ylabel(r"Relative $L_2$ error [%]")
            correlation_axis.set_ylabel("Pearson correlation")
            correlation_axis.set_ylim(1, 0)
            signal_axis.set_title(f"{title}: {field_name}")
            apply_standard_axes(signal_axis)

            # Use the Pearson-correlation axis for horizontal
            # gridlines. The signal axis retains the shared vertical
            # iteration gridlines.
            signal_axis.grid(False, which="both", axis="y")
            l2_axis.grid(False, which="both", axis="y")
            correlation_axis.minorticks_on()
            correlation_axis.grid(
                which="major",
                axis="y",
                linestyle="-",
                alpha=0.35,
            )
            correlation_axis.grid(
                which="minor",
                axis="y",
                linestyle=":",
                alpha=0.20,
            )
            correlation_axis.set_axisbelow(True)

            signal_axis.legend(
                handles,
                [handle.get_label() for handle in handles],
                loc="best",
                fontsize=9,
            )
            fig.subplots_adjust(right=0.80)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# FOM-RERUN COUNTER
# =============================================================================

def load_fom_rerun_counter(filename, active_range, delimiter=","):
    iterations, columns = load_error_signal_file(
        filename,
        active_range,
        delimiter,
    )
    if columns.shape[1] != 1:
        raise ValueError("FOM-rerun counter must contain exactly one value column.")

    values = columns[:, 0]
    rounded = np.rint(values)
    if not np.all(np.isfinite(values)) or not np.allclose(
        values,
        rounded,
        atol=1.0e-12,
    ):
        raise ValueError("FOM-rerun counter values must be finite integers.")

    counter = rounded.astype(np.int64)
    if np.any(counter < 0) or np.any(np.diff(counter) < 0):
        raise ValueError("FOM-rerun counter must be nonnegative and nondecreasing.")

    changes = np.diff(counter, prepend=counter[0])
    rerun_iterations = iterations[changes > 0]
    pattern = re.compile(
        r"^\s*#\s*threshold_exceeded,iteration=([^,]+),"
        r"variables=([^,]*),mask=([01](?:\|[01])*)\s*$"
    )
    records = {}

    with Path(filename).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if not match:
                continue

            iteration = float(match.group(1))
            key = int(np.rint(iteration))
            if not np.isclose(iteration, key, atol=1.0e-9):
                raise ValueError(
                    f"Noninteger trigger iteration on line {line_number}."
                )

            mask = np.array(
                [item == "1" for item in match.group(3).split("|")]
            )
            old = records.get(key, np.zeros(len(mask), bool))
            size = max(len(old), len(mask))
            old = np.pad(old, (0, size - len(old)))
            mask = np.pad(mask, (0, size - len(mask)))
            records[key] = old | mask

    n_variables = max(
        len(CONSERVATIVE_FIELD_NAMES),
        max((len(mask) for mask in records.values()), default=0),
    )
    per_variable = [[] for _ in range(n_variables)]
    selected_min, selected_max = iterations[0], iterations[-1]

    for key, mask in sorted(records.items()):
        if selected_min <= key <= selected_max:
            for i, exceeded in enumerate(mask):
                if exceeded:
                    per_variable[i].append(float(key))

    return (
        iterations,
        counter,
        rerun_iterations,
        tuple(np.asarray(values) for values in per_variable),
    )


def reruns_per_window(iterations, counter, window):
    require_positive(window, "FOM rerun window")
    x = np.asarray(iterations, float)
    counter = np.asarray(counter, np.int64)
    if x.ndim != 1 or x.shape != counter.shape or not len(x):
        raise ValueError("Counter arrays must be nonempty, 1-D, and matching.")

    increments = np.diff(counter, prepend=counter[0])
    prefix = np.r_[0, np.cumsum(increments)]
    result = np.full(x.shape, np.nan)

    for i, current in enumerate(x):
        if current - x[0] >= window:
            first = np.searchsorted(x, current - window, side="right")
            result[i] = prefix[i + 1] - prefix[first]
    return result


def plot_fom_rerun_history(iterations, counter, pdf, window):
    recent = reruns_per_window(iterations, counter, window)
    fig, axis = plt.subplots(figsize=(12, 7))
    recent_axis = axis.twinx()

    cumulative, = axis.step(
        iterations,
        counter,
        where="post",
        color="tab:blue",
        linewidth=2,
        label="Cumulative FOM reruns",
    )
    recent_line, = recent_axis.plot(
        iterations,
        recent,
        color="tab:purple",
        linewidth=2,
        label=f"Reruns in previous {window:g} iterations",
    )
    axis.set_ylabel("Cumulative FOM reruns")
    recent_axis.set_ylabel("Recent FOM reruns")
    axis.set_ylim(bottom=0)
    recent_axis.set_ylim(bottom=0)
    axis.set_title("Conservation-Error FOM Rerun History")
    apply_standard_axes(axis)
    axis.legend(handles=[cumulative, recent_line], loc="best")
    save_figure(pdf, fig)


# =============================================================================
# PROGRAM ORCHESTRATION
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate selected FOM/ROM conservation-error PDFs."
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

    base_output = (
        args.output.expanduser().resolve()
        if args.output
        else rom_dir / "FOM_ROM_metrics.pdf"
    )
    pdf_filename = output_filename(base_output, active_range)
    pdf_filename.parent.mkdir(parents=True, exist_ok=True)

    print_section(
        "Run configuration",
        fom_directory=fom_dir,
        rom_directory=rom_dir,
        output_pdf=pdf_filename,
        iteration_range=active_range,
        enabled_plots=[key for key, enabled in PLOT_ENABLED.items() if enabled],
    )

    fom_primitive, rom_primitive, primitive_iterations = load_common_npy_data(
        fom_dir,
        rom_dir,
        PRIMITIVE_FILE_PATTERN,
        active_range,
        "primitive",
    )
    relative_l2, correlation = compute_metric_histories(
        fom_primitive,
        rom_primitive,
    )
    del fom_primitive, rom_primitive

    conservative_iterations = None
    fom_integrals = None
    rom_integrals = None
    if is_plot_enabled("conservative_integrals"):
        conservative_iterations = primitive_iterations
        fom_cons, rom_cons = load_conservative_data_for_iterations(
            fom_dir,
            rom_dir,
            conservative_iterations,
        )
        fom_integrals, rom_integrals = compute_conservative_integrals(
            fom_cons,
            rom_cons,
        )
        del fom_cons, rom_cons

    rom_error_dir = rom_dir.parent / "error"
    counter_file = rom_error_dir / "error_cons_fom_iter_counter.txt"
    counter_iterations = None
    counter_values = None
    _rerun_iterations = np.array([])
    reruns_by_variable = tuple(
        np.array([]) for _ in range(len(CONSERVATIVE_FIELD_NAMES))
    )

    if counter_file.is_file():
        try:
            (
                counter_iterations,
                counter_values,
                _rerun_iterations,
                reruns_by_variable,
            ) = load_fom_rerun_counter(
                counter_file,
                active_range,
                args.error_delimiter,
            )
        except (OSError, ValueError) as error:
            print(f"Warning: ignoring invalid counter file {counter_file}: {error}")

    with PdfPages(pdf_filename) as pdf:
        if is_plot_enabled("sampling_frequency"):
            plot_sampling_frequency(
                rom_error_dir, pdf, active_range, args.error_delimiter,
            )

        if is_plot_enabled("fom_rerun_history"):
            if counter_iterations is None:
                print(f"Warning: FOM-rerun counter file unavailable: {counter_file}")
            else:
                plot_fom_rerun_history(
                    counter_iterations,
                    counter_values,
                    pdf,
                    args.fom_rerun_window,
                )

        if is_plot_enabled("primitive_metrics"):
            plot_primitive_metrics(
                primitive_iterations,
                relative_l2,
                correlation,
                PRIMITIVE_FIELD_NAMES,
                pdf,
            )

        if is_plot_enabled("conservative_integrals"):
            plot_conservative_integrals(
                conservative_iterations,
                fom_integrals,
                rom_integrals,
                pdf,
            )

        if any(
            is_plot_enabled(key)
            for key in (
                "rom_conservation_error_log",
                "rom_conservation_error_linear",
                "winsorized_mean_comparison",
                "histogram_density_band",
                "saved_short_mad_moving_average",
                "saved_first_mad_moving_average",
                "rolling_cv",
                "conservation_error_by_phase",
                "conservation_error_phase_mad",
                "conservation_error_phase_quantiles",
                "conservation_error_phase_tail_body",
                "phase_mad_ratio_online",
                "phase_quarter_mad_ratio_online",
                "signed_conservation_error_cycle_sum",
            )
        ):
            plot_rom_conservation_errors(
                rom_error_dir,
                pdf,
                active_range,
                reruns_by_variable,
                args.error_delimiter,
            )

        if is_plot_enabled("interpolation_error_signals"):
            plot_interpolation_error_signals(
                primitive_iterations,
                relative_l2,
                correlation,
                PRIMITIVE_FIELD_NAMES,
                rom_error_dir,
                pdf,
                active_range,
                _rerun_iterations,
                args.error_short_window,
                args.error_long_window,
                args.error_delimiter,
            )

    print(f"Saved: {pdf_filename}", flush=True)


def main():
    args = parse_arguments()
    validate_configuration()
    require_positive(args.error_short_window, "--error-short-window")
    require_positive(args.error_long_window, "--error-long-window")
    require_positive(args.fom_rerun_window, "--fom-rerun-window")

    for index, active_range in enumerate(configured_ranges()):
        active_range = validate_iteration_range(active_range, index)
        run_range(args, active_range)


if __name__ == "__main__":
    main()
