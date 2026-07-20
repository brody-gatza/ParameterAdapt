#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field, replace
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# USER SETTINGS
# ============================================================

@dataclass(frozen=True)
class Settings:
    # ----------------------------
    # Input / output
    # ----------------------------
    delimiter: str = ","
    skiprows: int = 0

    output_dir: Path = Path("figures")
    pdf_output_file: str = "all_comparisons.pdf"

    show_grid: bool = True
    show_figures: bool = False

    # ----------------------------
    # Iteration filtering
    # ----------------------------
    min_iteration_to_include: float | None = None
    max_iteration_to_include: float | None = None

    # ----------------------------
    # Plot simplification
    # ----------------------------
    plot_every_nth_point: int | None = 33

    # ----------------------------
    # Moving/running averages
    # ----------------------------
    plot_moving_average: bool = True
    moving_average_windows: list[int] = field(default_factory=lambda: [1000])
    baseline_moving_average_window: int = 5000

    plot_running_average: bool = True

    # ----------------------------
    # Main derived figure toggles
    # ----------------------------
    plot_online_cumulative_sum_of_moving_average_comparison: bool = True
    plot_slope_of_cumulative_sum_of_moving_average_comparison: bool = True

    # Disabled per request.
    plot_raw_ma_and_cumulative_sum_second_axis: bool = False

    plot_moving_average_ratio: bool = False
    plot_rms_ratio: bool = False
    plot_lag1_autocorrelation: bool = True
    plot_short_long_rms: bool = False

    # ----------------------------
    # Slope figure toggles
    # ----------------------------
    calculate_slope: bool = True

    plot_slope_of_raw: bool = True
    plot_slope_of_moving_average: bool = True
    plot_slope_of_rms_average: bool = False
    plot_slope_magnitude: bool = False
    plot_moving_average_of_slope: bool = True
    plot_raw_slope_moving_average_only: bool = False
    plot_positive_raw_slopes_only: bool = False

    plot_cumulative_sum_of_error_slope: bool = False
    plot_cumulative_sum_of_moving_average_slope: bool = False
    plot_moving_average_of_cumulative_sum: bool = False

    plot_raw_slope_sign_running_total: bool = True
    plot_raw_slope_sign_indicator: bool = False
    plot_moving_average_of_raw_slope_sign_indicator: bool = False



    slope_logy: bool = False
    slope_magnitude_logy: bool = False
    cumulative_slope_logy: bool = False

    # ----------------------------
    # Ratio plot options
    # ----------------------------
    ratio_logy: bool = False

    # ----------------------------
    # Y-axis limits
    # ----------------------------
    apply_auto_y_limits: bool = True

    y_limit_padding_fraction: float = 0.08
    auto_y_limits_ignore_extreme_outliers: bool = True
    y_limit_log_min_positive_value: float | None = None

    auto_y_limits_endpoint_exclusion_fraction: float = 0.1
    auto_y_limits_endpoint_exclusion_max_points: int | None = None

    # ----------------------------
    # Colors
    # ----------------------------
    primary_colors: list[str] = field(
        default_factory=lambda: [
            "blue",
            "orange",
            "green",
            "purple",
        ]
    )

    secondary_colors: list[str] = field(
        default_factory=lambda: [
            "cyan",
            "red",
            "lime",
            "magenta",
        ]
    )

    tertiary_colors: list[str] = field(
        default_factory=lambda: [
            "navy",
            "darkorange",
            "darkgreen",
            "indigo",
        ]
    )

    # ----------------------------
    # Variable names
    # ----------------------------
    primitive_variable_names: list[str] | None = None
    conservative_variable_names: list[str] | None = None

    # ----------------------------
    # Variable normalization
    # ----------------------------
    normalize_variables: bool = False

    # primitive_variable_normalization_factors: list[float] = field(
    #     default_factory=lambda: [
    #         2.55,
    #         1250,
    #         3.46E6,
    #         3.52E3,
    #         1.0,
    #         3.86E12,
    #     ]
    # )

    # conservative_variable_normalization_factors: list[float] = field(
    #     default_factory=lambda: [
    #         3.67E-4,
    #         1.38E-1,
    #         2.51E3,
    #         2.19E-4,
    #     ]
    # )


# ----------------------------
# Data Locations
# ----------------------------

@dataclass(frozen=True)
class FigureGroup:
    title: str
    filenames: list[str]
    selected_vars: list[int] | None = None
    logy: bool = True
    skiprows: int | None = None


SETTINGS = Settings(
    primitive_variable_names=[
        "Density",
        "Velocity",
        "Pressure",
        "Temperature",
        "Y Reactant",
        "Heat Release Rate",
    ],
)

FIGURE_GROUPS = [
    FigureGroup(
        title="Primitive Variables - Max",
        filenames=[
            "prim_interp_max.txt",
            # "prim_proj_max.txt",
        ],
        selected_vars=None,
        logy=True,
    ),
    FigureGroup(
        title="Conservative Variables - Max",
        filenames=[
            "cons_interp_max.txt",
            # "cons_proj_max.txt",
        ],
        selected_vars=None,
        logy=True,
    ),
    FigureGroup(
        title="Primitive Variables - Avg",
        filenames=[
            "prim_interp_avg.txt",
            # "prim_proj_avg.txt",
        ],
        selected_vars=None,
        logy=True,
    ),
    FigureGroup(
        title="Conservative Variables - Avg",
        filenames=[
            "cons_interp_avg.txt",
            # "cons_proj_avg.txt",
        ],
        selected_vars=None,
        logy=True,
    ),
]


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class LoadedDataset:
    filename: Path
    iterations: np.ndarray
    variables: np.ndarray


@dataclass
class LoadedGroup:
    group: FigureGroup
    datasets: list[LoadedDataset]
    variable_indices: list[int]


# ============================================================
# VARIABLE NAME HELPERS
# ============================================================

def get_variable_kind_for_filename(filename: Path) -> str | None:
    filename_text = filename.name.lower()

    if "prim" in filename_text:
        return "primitive"

    if "cons" in filename_text:
        return "conservative"

    return None


def get_variable_kind_for_group(group: FigureGroup) -> str | None:
    title_text = group.title.lower()

    if "primitive" in title_text or "prim" in title_text:
        return "primitive"

    if "conservative" in title_text or "cons" in title_text:
        return "conservative"

    detected_kinds = set()

    for filename_text in group.filenames:
        variable_kind = get_variable_kind_for_filename(Path(filename_text))

        if variable_kind is not None:
            detected_kinds.add(variable_kind)

    if len(detected_kinds) == 1:
        return detected_kinds.pop()

    return None


def get_variable_names_for_kind(
    variable_kind: str | None,
    settings: Settings,
) -> list[str] | None:
    if variable_kind == "primitive":
        return settings.primitive_variable_names

    if variable_kind == "conservative":
        return settings.conservative_variable_names

    return None


def get_default_variable_label(var_index: int) -> str:
    return f"var{var_index + 1}"


def get_variable_label_from_names(
    variable_names: list[str] | None,
    var_index: int,
) -> str:
    default_label = get_default_variable_label(var_index)

    if variable_names is None:
        return default_label

    if var_index >= len(variable_names):
        return default_label

    variable_label = str(variable_names[var_index]).strip()

    if variable_label == "":
        return default_label

    return variable_label


def get_variable_label_for_filename(
    filename: Path,
    settings: Settings,
    var_index: int,
) -> str:
    variable_kind = get_variable_kind_for_filename(filename)
    variable_names = get_variable_names_for_kind(variable_kind, settings)

    return get_variable_label_from_names(
        variable_names=variable_names,
        var_index=var_index,
    )


def get_variable_label_for_group(
    group: FigureGroup,
    settings: Settings,
    var_index: int,
) -> str:
    variable_kind = get_variable_kind_for_group(group)
    variable_names = get_variable_names_for_kind(variable_kind, settings)

    return get_variable_label_from_names(
        variable_names=variable_names,
        var_index=var_index,
    )


# ============================================================
# DATA LOADING
# ============================================================

def apply_iteration_limits(
    iterations: np.ndarray,
    variables: np.ndarray,
    filename: Path,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    iterations = np.asarray(iterations, dtype=float)
    variables = np.asarray(variables, dtype=float)

    keep_mask = np.ones_like(iterations, dtype=bool)

    if settings.min_iteration_to_include is not None:
        keep_mask &= iterations >= settings.min_iteration_to_include

    if settings.max_iteration_to_include is not None:
        keep_mask &= iterations <= settings.max_iteration_to_include

    iterations = iterations[keep_mask]
    variables = variables[keep_mask, :]

    if len(iterations) == 0:
        raise ValueError(
            f"No data remains in {filename} after applying iteration limits: "
            f"min_iteration_to_include = {settings.min_iteration_to_include}, "
            f"max_iteration_to_include = {settings.max_iteration_to_include}."
        )

    return iterations, variables


def get_normalization_factors_for_filename(
    filename: Path,
    settings: Settings,
) -> list:
    filename_text = filename.name.lower()

    if "prim" in filename_text:
        return settings.primitive_variable_normalization_factors

    if "cons" in filename_text:
        return settings.conservative_variable_normalization_factors

    return []


def normalize_loaded_variables(
    variables: np.ndarray,
    filename: Path,
    settings: Settings,
) -> np.ndarray:
    variables = np.asarray(variables, dtype=float).copy()

    if not settings.normalize_variables:
        return variables

    normalization_factors = get_normalization_factors_for_filename(
        filename=filename,
        settings=settings,
    )

    if not normalization_factors:
        return variables

    num_variables = variables.shape[1]

    for var_index in range(num_variables):
        variable_label = get_variable_label_for_filename(
            filename=filename,
            settings=settings,
            var_index=var_index,
        )

        if var_index >= len(normalization_factors):
            normalization_factor = 1.0
        else:
            normalization_factor = float(normalization_factors[var_index])

        if not np.isfinite(normalization_factor):
            raise ValueError(
                f"Normalization factor for {variable_label} in "
                f"{filename} is not finite."
            )

        if normalization_factor == 0.0:
            raise ValueError(
                f"Normalization factor for {variable_label} in "
                f"{filename} is zero."
            )

        variables[:, var_index] = variables[:, var_index] / normalization_factor

    return variables


def load_dataset(
    filename: Path,
    settings: Settings,
    skiprows: int | None = None,
) -> LoadedDataset:
    actual_skiprows = settings.skiprows if skiprows is None else skiprows

    data = np.loadtxt(
        filename,
        delimiter=settings.delimiter,
        skiprows=actual_skiprows,
    )

    iterations = data[:, 0]
    variables = data[:, 1:]

    iterations, variables = apply_iteration_limits(
        iterations,
        variables,
        filename,
        settings,
    )

    variables = normalize_loaded_variables(
        variables=variables,
        filename=filename,
        settings=settings,
    )

    return LoadedDataset(
        filename=filename,
        iterations=iterations,
        variables=variables,
    )


def load_group_data(
    group: FigureGroup,
    settings: Settings,
) -> LoadedGroup:
    datasets: list[LoadedDataset] = []
    max_num_vars = 0

    for filename_text in group.filenames:
        dataset = load_dataset(
            filename=Path(filename_text),
            settings=settings,
            skiprows=group.skiprows,
        )

        datasets.append(dataset)
        max_num_vars = max(max_num_vars, dataset.variables.shape[1])

    if group.selected_vars is None:
        variable_indices = list(range(max_num_vars))
    else:
        variable_indices = [index - 1 for index in group.selected_vars]

    return LoadedGroup(
        group=group,
        datasets=datasets,
        variable_indices=variable_indices,
    )


# ============================================================
# COMPUTATION HELPERS
# ============================================================

def downsample_for_plotting(
    x: np.ndarray,
    y: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    every_nth_point = settings.plot_every_nth_point

    if every_nth_point is None:
        return x, y

    if every_nth_point <= 1:
        return x, y

    return x[::every_nth_point], y[::every_nth_point]


def compute_moving_average(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def compute_moving_average_ignore_nan(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    return x, y_avg


def compute_moving_rms(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window <= 0:
        raise ValueError("Moving-RMS window must be positive.")

    y_rms = np.full(len(y), np.nan)
    cumulative_sum_squares = 0.0

    for i in range(len(y)):
        cumulative_sum_squares += y[i] * y[i]

        if i >= window:
            cumulative_sum_squares -= y[i - window] * y[i - window]
            y_rms[i] = np.sqrt(cumulative_sum_squares / window)
        else:
            y_rms[i] = np.sqrt(cumulative_sum_squares / (i + 1))

    return x, y_rms


def compute_moving_lag1_autocorrelation(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window <= 0:
        raise ValueError("Lag-1 autocorrelation window must be positive.")

    lag1_autocorrelation = np.full(len(y), np.nan)

    for i in range(len(y)):
        start_index = max(0, i - window + 1)
        y_window = y[start_index : i + 1]

        if len(y_window) < 3:
            continue

        y0 = y_window[:-1]
        y1 = y_window[1:]

        finite_mask = np.isfinite(y0) & np.isfinite(y1)

        if np.count_nonzero(finite_mask) < 2:
            continue

        y0 = y0[finite_mask]
        y1 = y1[finite_mask]

        y0 = y0 - np.mean(y0)
        y1 = y1 - np.mean(y1)

        denominator = np.sqrt(np.sum(y0 * y0) * np.sum(y1 * y1))

        if denominator == 0.0:
            lag1_autocorrelation[i] = 0.0
        else:
            lag1_autocorrelation[i] = np.sum(y0 * y1) / denominator

    return x, lag1_autocorrelation


def compute_running_average(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)

    cumulative_sum = np.cumsum(y)
    sample_count = np.arange(1, len(y) + 1)
    running_average = cumulative_sum / sample_count

    return x, running_average


def compute_slope_sign_indicator(
    slope_y: np.ndarray,
    use_nan_for_zero_or_invalid: bool = False,
) -> np.ndarray:
    slope_y = np.asarray(slope_y, dtype=float)

    if use_nan_for_zero_or_invalid:
        sign_indicator = np.full(len(slope_y), np.nan)
    else:
        sign_indicator = np.zeros(len(slope_y), dtype=float)

    positive_mask = np.isfinite(slope_y) & (slope_y > 0.0)
    negative_mask = np.isfinite(slope_y) & (slope_y < 0.0)

    sign_indicator[positive_mask] = 1.0
    sign_indicator[negative_mask] = -1.0

    return sign_indicator


def figure_raw_slope_sign_running_total(
    group: FigureGroup,
    settings: Settings,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            slope_x, raw_slope = compute_slope(x, y)

            slope_sign = compute_slope_sign_indicator(
                raw_slope,
                use_nan_for_zero_or_invalid=False,
            )

            slope_sign_running_total = np.cumsum(slope_sign)

            x_plot, running_total_plot = downsample_for_plotting(
                slope_x,
                slope_sign_running_total,
                settings,
            )

            ax.plot(
                x_plot,
                running_total_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.2,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} running total of raw slope sign"
                ),
                zorder=5,
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
            ylabel=f"running total sign d({variable_label})/dIteration",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    title = (
        f"{group.title} Running Total of Raw Slope Sign "
        "[+1 positive slope, -1 negative slope]"
    )

    finalize_figure(fig, axes, title)

    return fig


def compute_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def compute_cumulative_sum(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    return np.cumsum(y)


def compute_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    ratio = np.full(len(numerator), np.nan)

    valid_mask = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0.0)
    )

    ratio[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return ratio


def compute_cumsum_normalized_by_iteration(
    x: np.ndarray,
    cumulative_sum: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cumulative_sum = np.asarray(cumulative_sum, dtype=float)

    normalized = np.full(len(cumulative_sum), np.nan)

    valid_mask = (
        np.isfinite(x)
        & np.isfinite(cumulative_sum)
        & (x != 0.0)
    )

    normalized[valid_mask] = cumulative_sum[valid_mask] / x[valid_mask]

    return normalized


def compute_online_moving_average_comparison_cumulative_sum(
    x: np.ndarray,
    y: np.ndarray,
    moving_average_window: int,
    baseline_moving_average_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    _, y_short_ma = compute_moving_average(
        x,
        y,
        moving_average_window,
    )

    _, y_long_ma = compute_moving_average(
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

def get_endpoint_exclusion_count(
    num_points: int,
    settings: Settings,
) -> int:
    if num_points <= 0:
        return 0

    exclusion_fraction = settings.auto_y_limits_endpoint_exclusion_fraction

    if exclusion_fraction <= 0.0:
        return 0

    if exclusion_fraction >= 0.5:
        raise ValueError(
            "auto_y_limits_endpoint_exclusion_fraction must be less than 0.5."
        )

    exclusion_count = int(np.floor(exclusion_fraction * num_points))

    if settings.auto_y_limits_endpoint_exclusion_max_points is not None:
        exclusion_count = min(
            exclusion_count,
            settings.auto_y_limits_endpoint_exclusion_max_points,
        )

    max_safe_exclusion_count = max(0, (num_points - 2) // 2)
    exclusion_count = min(exclusion_count, max_safe_exclusion_count)

    return exclusion_count


def exclude_line_endpoints_for_y_limits(
    y: np.ndarray,
    settings: Settings,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return y

    exclusion_count = get_endpoint_exclusion_count(
        num_points=len(y),
        settings=settings,
    )

    if exclusion_count <= 0:
        return y

    if 2 * exclusion_count >= len(y):
        return y

    return y[exclusion_count:-exclusion_count]


def get_automatic_y_limits_from_data(
    y: np.ndarray,
) -> tuple[float, float] | None:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return None

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

    if y_min > y_max:
        return None

    return y_min, y_max


def pad_y_limits(
    visible_min: float,
    visible_max: float,
    settings: Settings,
    logy: bool = False,
    keep_zero_visible: bool = False,
) -> tuple[float, float] | None:
    if not np.isfinite(visible_min):
        return None

    if not np.isfinite(visible_max):
        return None

    if visible_min >= visible_max:
        return None

    padding_fraction = settings.y_limit_padding_fraction

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
    ax: Axes,
    settings: Settings,
    logy: bool = False,
    keep_zero_visible: bool = False,
) -> None:
    if not settings.auto_y_limits_ignore_extreme_outliers:
        return

    all_y_values_for_limits = []

    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]

        if len(y) == 0:
            continue

        if logy:
            y = y[y > 0.0]

            if settings.y_limit_log_min_positive_value is not None:
                y = y[y >= settings.y_limit_log_min_positive_value]

            if len(y) == 0:
                continue

            working_y = np.log10(y)
        else:
            working_y = y.copy()

        working_y = working_y[np.isfinite(working_y)]

        if len(working_y) == 0:
            continue

        working_y = exclude_line_endpoints_for_y_limits(
            y=working_y,
            settings=settings,
        )

        if len(working_y) > 0:
            all_y_values_for_limits.append(working_y)

    if not all_y_values_for_limits:
        return

    combined_y_values_for_limits = np.concatenate(all_y_values_for_limits)
    combined_y_values_for_limits = combined_y_values_for_limits[
        np.isfinite(combined_y_values_for_limits)
    ]

    if len(combined_y_values_for_limits) == 0:
        return

    automatic_limits = get_automatic_y_limits_from_data(
        combined_y_values_for_limits
    )

    if automatic_limits is None:
        return

    visible_min_working, visible_max_working = automatic_limits

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
        settings=settings,
        logy=logy,
        keep_zero_visible=keep_zero_visible,
    )

    if padded_limits is None:
        return

    final_min, final_max = padded_limits
    ax.set_ylim(final_min, final_max)


def apply_limits_if_enabled(
    ax: Axes,
    settings: Settings,
    logy: bool = False,
    keep_zero_visible: bool = False,
) -> None:
    if settings.apply_auto_y_limits:
        apply_adaptive_y_limits(
            ax=ax,
            settings=settings,
            logy=logy,
            keep_zero_visible=keep_zero_visible,
        )


# ============================================================
# PLOTTING STYLE HELPERS
# ============================================================

def get_plot_color(
    dataset_index: int,
    settings: Settings,
    color_role: str = "primary",
) -> str:
    if color_role == "primary":
        color_list = settings.primary_colors
    elif color_role == "secondary":
        color_list = settings.secondary_colors
    elif color_role == "tertiary":
        color_list = settings.tertiary_colors
    else:
        raise ValueError(
            "color_role must be 'primary', 'secondary', or 'tertiary'."
        )

    return color_list[dataset_index % len(color_list)]


def create_figure(variable_indices: list[int]):
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


def format_axis(
    ax: Axes,
    ylabel: str,
    settings: Settings,
    logy: bool = False,
) -> None:
    ax.set_ylabel(ylabel, fontsize=12)

    if logy:
        ax.set_yscale("log")

    ax.set_axisbelow(True)

    if settings.show_grid:
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


def finalize_figure(fig, axes, title: str) -> None:
    axes[-1].set_xlabel("Iteration", fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold")


def add_page_to_pdf(
    pdf: PdfPages,
    fig,
    settings: Settings,
) -> None:
    pdf.savefig(fig, bbox_inches="tight")

    if not settings.show_figures:
        plt.close(fig)


def merge_twin_axis_legends(
    ax_left: Axes,
    ax_right: Axes,
    fontsize: int = 10,
) -> None:
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
# FIGURE 1: MAIN DATA
# ============================================================

def figure_main_data(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
    plot_moving_average: bool,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            raw_linewidth = 1.0 if plot_moving_average else 2.0
            raw_alpha = 0.70 if plot_moving_average else 0.85

            x_plot, y_plot = downsample_for_plotting(x, y, settings)

            ax.plot(
                x_plot,
                y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=raw_linewidth,
                alpha=raw_alpha,
                label=dataset.filename.stem,
                zorder=3,
            )

            if plot_moving_average and moving_average_window is not None:
                x_avg, y_avg = compute_moving_average(
                    x,
                    y,
                    moving_average_window,
                )

                x_avg_plot, y_avg_plot = downsample_for_plotting(
                    x_avg,
                    y_avg,
                    settings,
                )

                ax.plot(
                    x_avg_plot,
                    y_avg_plot,
                    color=get_plot_color(dataset_index, settings, "secondary"),
                    linewidth=3.0,
                    alpha=1.0,
                    label=f"{dataset.filename.stem} CFD MA-{moving_average_window}",
                    zorder=5,
                )

            if settings.plot_running_average:
                x_running, y_running = compute_running_average(x, y)

                x_running_plot, y_running_plot = downsample_for_plotting(
                    x_running,
                    y_running,
                    settings,
                )

                ax.plot(
                    x_running_plot,
                    y_running_plot,
                    color=get_plot_color(dataset_index, settings, "tertiary"),
                    linewidth=2.4,
                    linestyle="-.",
                    alpha=0.95,
                    label=f"{dataset.filename.stem} running avg",
                    zorder=6,
                )

        format_axis(
            ax,
            ylabel=variable_label,
            settings=settings,
            logy=group.logy,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=group.logy,
            keep_zero_visible=False,
        )

    title = group.title

    if group.logy:
        title += " [log-y]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 2: CUMULATIVE MOVING-AVERAGE COMPARISON
# ============================================================

def figure_online_cumulative_ma_comparison(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax_right = ax_left.twinx()

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            (
                x_comparison,
                _y_short_ma,
                _y_long_ma,
                _moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=settings.baseline_moving_average_window,
            )

            normalized_cumsum = compute_cumsum_normalized_by_iteration(
                x_comparison,
                cumulative_moving_average_difference,
            )

            x_cumsum_plot, cumsum_plot = downsample_for_plotting(
                x_comparison,
                cumulative_moving_average_difference,
                settings,
            )

            x_norm_plot, norm_plot = downsample_for_plotting(
                x_comparison,
                normalized_cumsum,
                settings,
            )

            ax_left.plot(
                x_cumsum_plot,
                cumsum_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.4,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} cumsum "
                    f"(MA-{moving_average_window} - "
                    f"MA-{settings.baseline_moving_average_window})"
                ),
                zorder=5,
            )

            ax_right.plot(
                x_norm_plot,
                norm_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.4,
                linestyle="-.",
                alpha=1.0,
                label=(
                    f"{dataset.filename.stem} cumsum / iteration "
                    f"(MA-{moving_average_window} - "
                    f"MA-{settings.baseline_moving_average_window})"
                ),
                zorder=6,
            )

        ax_left.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        ax_right.axhline(
            0.0,
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            zorder=1,
        )

        ax_left.set_ylabel(
            f"cumsum MA difference {variable_label}",
            fontsize=12,
        )

        ax_right.set_ylabel(
            f"cumsum MA difference / iteration {variable_label}",
            fontsize=12,
        )

        ax_left.set_axisbelow(True)

        if settings.show_grid:
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

        merge_twin_axis_legends(
            ax_left,
            ax_right,
            fontsize=9,
        )

        apply_limits_if_enabled(
            ax_left,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

        apply_limits_if_enabled(
            ax_right,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    title = (
        f"{group.title} Cumulative Moving-Average Difference "
        f"and Iteration-Normalized Cumulative Difference "
        f"[MA-{moving_average_window} minus "
        f"MA-{settings.baseline_moving_average_window}]"
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 3: SLOPE OF CUMULATIVE MOVING-AVERAGE COMPARISON
# ============================================================

def figure_slope_of_cumulative_ma_comparison(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            (
                x_comparison,
                _y_short_ma,
                _y_long_ma,
                _moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=settings.baseline_moving_average_window,
            )

            slope_x, slope_y = compute_slope(
                x_comparison,
                cumulative_moving_average_difference,
            )

            x_slope_plot, slope_y_plot = downsample_for_plotting(
                slope_x,
                slope_y,
                settings,
            )

            ax.plot(
                x_slope_plot,
                slope_y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.2,
                alpha=0.55,
                label=(
                    f"{dataset.filename.stem} slope of cumsum "
                    f"(MA-{moving_average_window} - "
                    f"MA-{settings.baseline_moving_average_window})"
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
                settings,
            )

            ax.plot(
                x_slope_avg_plot,
                slope_avg_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=3.0,
                alpha=1.0,
                label=(
                    f"{dataset.filename.stem} slope of cumsum "
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
            ylabel=f"d cumsum MA comparison ({variable_label})/dIteration",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    title = (
        f"{group.title} Slope of Cumulative Moving-Average Comparison "
        f"[MA-{moving_average_window} minus "
        f"MA-{settings.baseline_moving_average_window}] "
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 4: RAW, MOVING AVERAGE, AND CUMSUM ON SECOND AXIS
# Disabled by default, retained for optional future use.
# ============================================================

def figure_raw_ma_cumsum_second_axis(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax_right = ax_left.twinx()

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            (
                x_comparison,
                y_short_ma,
                _y_long_ma,
                _moving_average_difference,
                cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=settings.baseline_moving_average_window,
            )

            x_raw_plot, y_raw_plot = downsample_for_plotting(x, y, settings)

            x_short_plot, y_short_plot = downsample_for_plotting(
                x_comparison,
                y_short_ma,
                settings,
            )

            x_cumsum_plot, cumsum_plot = downsample_for_plotting(
                x_comparison,
                cumulative_moving_average_difference,
                settings,
            )

            ax_left.plot(
                x_raw_plot,
                y_raw_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.0,
                alpha=0.55,
                label=f"{dataset.filename.stem} raw",
                zorder=3,
            )

            ax_left.plot(
                x_short_plot,
                y_short_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.6,
                alpha=1.0,
                label=f"{dataset.filename.stem} CFD MA-{moving_average_window}",
                zorder=5,
            )

            ax_right.plot(
                x_cumsum_plot,
                cumsum_plot,
                color=get_plot_color(dataset_index, settings, "tertiary"),
                linewidth=2.6,
                linestyle="-.",
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} cumsum "
                    f"(MA-{moving_average_window} - "
                    f"MA-{settings.baseline_moving_average_window})"
                ),
                zorder=6,
            )

        ax_left.set_ylabel(variable_label, fontsize=12)
        ax_right.set_ylabel(
            f"cumsum(MA-{moving_average_window} - "
            f"{settings.baseline_moving_average_window})",
            fontsize=12,
        )

        if group.logy:
            ax_left.set_yscale("log")

        ax_left.set_axisbelow(True)

        if settings.show_grid:
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
            settings=settings,
            logy=group.logy,
            keep_zero_visible=False,
        )

        apply_limits_if_enabled(
            ax_right,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    title = (
        f"{group.title} Raw Data, MA, and Cumulative MA Comparison "
        f"[MA-{moving_average_window} vs "
        f"MA-{settings.baseline_moving_average_window}]"
    )

    if group.logy:
        title += " [left axis log-y]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 5: MOVING-AVERAGE RATIO
# ============================================================

def figure_moving_average_ratio(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            _, y_short_ma = compute_moving_average(
                x,
                y,
                moving_average_window,
            )

            _, y_long_ma = compute_moving_average(
                x,
                y,
                settings.baseline_moving_average_window,
            )

            ma_ratio = compute_ratio(y_short_ma, y_long_ma)

            x_ratio_plot, ratio_plot = downsample_for_plotting(
                x,
                ma_ratio,
                settings,
            )

            ax.plot(
                x_ratio_plot,
                ratio_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.2,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} "
                    f"MA-{moving_average_window}/"
                    f"MA-{settings.baseline_moving_average_window}"
                ),
                zorder=5,
            )

        ax.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=1,
        )

        format_axis(
            ax,
            ylabel=f"MA ratio ({variable_label})",
            settings=settings,
            logy=settings.ratio_logy,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=settings.ratio_logy,
            keep_zero_visible=False,
        )

    title = (
        f"{group.title} Moving-Average Ratio "
        f"[MA-{moving_average_window} / "
        f"MA-{settings.baseline_moving_average_window}]"
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 6: RMS RATIO
# ============================================================

def figure_rms_ratio(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            _, short_rms = compute_moving_rms(
                x,
                y,
                moving_average_window,
            )

            _, long_rms = compute_moving_rms(
                x,
                y,
                settings.baseline_moving_average_window,
            )

            rms_ratio = compute_ratio(short_rms, long_rms)

            x_ratio_plot, ratio_plot = downsample_for_plotting(
                x,
                rms_ratio,
                settings,
            )

            ax.plot(
                x_ratio_plot,
                ratio_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.2,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} "
                    f"RMS-{moving_average_window}/"
                    f"RMS-{settings.baseline_moving_average_window}"
                ),
                zorder=5,
            )

        ax.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=1,
        )

        format_axis(
            ax,
            ylabel=f"RMS ratio ({variable_label})",
            settings=settings,
            logy=settings.ratio_logy,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=settings.ratio_logy,
            keep_zero_visible=False,
        )

    title = (
        f"{group.title} RMS Ratio "
        f"[RMS-{moving_average_window} / "
        f"RMS-{settings.baseline_moving_average_window}]"
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 7: LAG-1 AUTOCORRELATION
# ============================================================

def figure_lag1_autocorrelation(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_lag1, lag1_autocorrelation = compute_moving_lag1_autocorrelation(
                x,
                y,
                moving_average_window,
            )

            x_lag1_plot, lag1_autocorrelation_plot = downsample_for_plotting(
                x_lag1,
                lag1_autocorrelation,
                settings,
            )

            ax.plot(
                x_lag1_plot,
                lag1_autocorrelation_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.0,
                alpha=0.90,
                label=(
                    f"{dataset.filename.stem} lag-1 autocorr "
                    f"window-{moving_average_window}"
                ),
                zorder=5,
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
            ylabel=f"lag-1 autocorr ({variable_label})",
            settings=settings,
            logy=False,
        )

        ax.set_ylim(-1.05, 1.05)

    title = (
        f"{group.title} Lag-1 Autocorrelation "
        f"[window = {moving_average_window}]"
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 8: SHORT AND LONG RMS
# ============================================================

def figure_short_long_rms(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_short_rms, short_rms = compute_moving_rms(
                x,
                y,
                moving_average_window,
            )

            x_long_rms, long_rms = compute_moving_rms(
                x,
                y,
                settings.baseline_moving_average_window,
            )

            x_short_rms_plot, short_rms_plot = downsample_for_plotting(
                x_short_rms,
                short_rms,
                settings,
            )

            x_long_rms_plot, long_rms_plot = downsample_for_plotting(
                x_long_rms,
                long_rms,
                settings,
            )

            ax.plot(
                x_short_rms_plot,
                short_rms_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.6,
                alpha=1.0,
                label=f"{dataset.filename.stem} RMS-{moving_average_window}",
                zorder=6,
            )

            ax.plot(
                x_long_rms_plot,
                long_rms_plot,
                color=get_plot_color(dataset_index, settings, "tertiary"),
                linewidth=2.6,
                linestyle="-.",
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} RMS-"
                    f"{settings.baseline_moving_average_window}"
                ),
                zorder=5,
            )

        format_axis(
            ax,
            ylabel=f"RMS ({variable_label})",
            settings=settings,
            logy=group.logy,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=group.logy,
            keep_zero_visible=False,
        )

    title = (
        f"{group.title} Short and Long RMS "
        f"[RMS-{moving_average_window} vs "
        f"RMS-{settings.baseline_moving_average_window}]"
    )

    if group.logy:
        title += " [log-y]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 9: SLOPE
# ============================================================

def figure_slope(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
    slope_source: str,
    plot_magnitude: bool,
    plot_positive_only: bool,
    plot_slope_curve: bool,
    plot_moving_average_of_slope: bool,
    slope_logy: bool,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    use_second_axis_for_moving_average_of_slope = (
        plot_slope_curve
        and plot_moving_average_of_slope
        and moving_average_window is not None
        and slope_source in ("raw", "ma")
    )

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        if use_second_axis_for_moving_average_of_slope:
            ax_right = ax_left.twinx()
        else:
            ax_right = None

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            if slope_source == "raw":
                slope_x, slope_y = compute_slope(x, y)
                source_label = "raw slope"

            elif slope_source == "ma":
                if moving_average_window is None:
                    print(
                        f"Warning: moving_average_window is None for "
                        f"{dataset.filename}. Skipping slope of moving average."
                    )
                    continue

                x_avg, y_avg = compute_moving_average(
                    x,
                    y,
                    moving_average_window,
                )

                slope_x, slope_y = compute_slope(x_avg, y_avg)
                source_label = f"CFD MA-{moving_average_window} slope"

            else:
                raise ValueError("slope_source must be either 'raw' or 'ma'.")

            if plot_magnitude:
                slope_y = np.abs(slope_y)

                line_label = f"{dataset.filename.stem} |{source_label}|"

                ma_label = (
                    f"{dataset.filename.stem} moving avg of |{source_label}| "
                    f"CFD MA-{moving_average_window}"
                )

            elif plot_positive_only:
                slope_y = np.where(slope_y > 0.0, slope_y, np.nan)

                line_label = f"{dataset.filename.stem} positive {source_label}"

                ma_label = (
                    f"{dataset.filename.stem} moving avg of positive "
                    f"{source_label} CFD MA-{moving_average_window}"
                )

            else:
                line_label = f"{dataset.filename.stem} {source_label}"

                ma_label = (
                    f"{dataset.filename.stem} moving avg of {source_label} "
                    f"CFD MA-{moving_average_window}"
                )

            if plot_slope_curve:
                slope_x_plot, slope_y_plot = downsample_for_plotting(
                    slope_x,
                    slope_y,
                    settings,
                )

                ax_left.plot(
                    slope_x_plot,
                    slope_y_plot,
                    color=get_plot_color(dataset_index, settings, "primary"),
                    linewidth=1.2,
                    alpha=0.55,
                    label=line_label,
                    zorder=3,
                )

            if plot_moving_average_of_slope and moving_average_window is not None:
                x_slope_avg, slope_avg = compute_moving_average_ignore_nan(
                    slope_x,
                    slope_y,
                    moving_average_window,
                )

                x_slope_avg_plot, slope_avg_plot = downsample_for_plotting(
                    x_slope_avg,
                    slope_avg,
                    settings,
                )

                if ax_right is not None:
                    slope_average_axis = ax_right
                    slope_average_zorder = 7
                else:
                    slope_average_axis = ax_left
                    slope_average_zorder = 6

                slope_average_axis.plot(
                    x_slope_avg_plot,
                    slope_avg_plot,
                    color=get_plot_color(dataset_index, settings, "secondary"),
                    linewidth=3.0,
                    alpha=1.0,
                    linestyle="-",
                    label=ma_label,
                    zorder=slope_average_zorder,
                )

        if not plot_magnitude and not plot_positive_only and plot_slope_curve:
            ax_left.axhline(
                0.0,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.5,
                zorder=1,
            )

            if ax_right is not None:
                ax_right.axhline(
                    0.0,
                    color="gray",
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.5,
                    zorder=1,
                )

        if plot_magnitude:
            if slope_source == "raw":
                ylabel_left = f"|d({variable_label})/dIteration|"
                ylabel_right = (
                    f"moving avg of |d({variable_label})/dIteration|"
                )
            else:
                ylabel_left = f"|d(CFD MA-{moving_average_window} {variable_label})/dIteration|"
                ylabel_right = (
                    f"moving avg of |d(CFD MA-{moving_average_window} "
                    f"{variable_label})/dIteration|"
                )

        elif plot_positive_only:
            if slope_source == "raw":
                ylabel_left = f"positive d({variable_label})/dIteration"
                ylabel_right = (
                    f"moving avg of positive d({variable_label})/dIteration"
                )
            else:
                ylabel_left = (
                    f"positive d(CFD MA-{moving_average_window} "
                    f"{variable_label})/dIteration"
                )
                ylabel_right = (
                    f"moving avg of positive d(CFD MA-{moving_average_window} "
                    f"{variable_label})/dIteration"
                )

        else:
            if slope_source == "raw":
                ylabel_left = f"d({variable_label})/dIteration"
                ylabel_right = (
                    f"moving avg of d({variable_label})/dIteration"
                )
            else:
                ylabel_left = (
                    f"d(CFD MA-{moving_average_window} "
                    f"{variable_label})/dIteration"
                )
                ylabel_right = (
                    f"moving avg of d(CFD MA-{moving_average_window} "
                    f"{variable_label})/dIteration"
                )

        if ax_right is None:
            format_axis(
                ax_left,
                ylabel=ylabel_left,
                settings=settings,
                logy=slope_logy,
            )

            apply_limits_if_enabled(
                ax_left,
                settings=settings,
                logy=slope_logy,
                keep_zero_visible=not slope_logy,
            )

        else:
            ax_left.set_ylabel(ylabel_left, fontsize=12)
            ax_right.set_ylabel(ylabel_right, fontsize=12)

            if slope_logy:
                ax_left.set_yscale("log")
                ax_right.set_yscale("log")

            ax_left.set_axisbelow(True)

            if settings.show_grid:
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

            merge_twin_axis_legends(
                ax_left,
                ax_right,
                fontsize=9,
            )

            apply_limits_if_enabled(
                ax_left,
                settings=settings,
                logy=slope_logy,
                keep_zero_visible=not slope_logy,
            )

            apply_limits_if_enabled(
                ax_right,
                settings=settings,
                logy=slope_logy,
                keep_zero_visible=not slope_logy,
            )

    title = group.title

    if slope_source == "raw":
        if plot_magnitude:
            title += " Slope Magnitude [|raw slope|]"
        elif plot_positive_only:
            title += " Positive Raw Slopes Only"
        elif not plot_slope_curve:
            title += (
                f" Raw Slope CFD Moving Average Only "
                f"[MA window = {moving_average_window}]"
            )
        else:
            title += " Slope [raw slope]"

    elif slope_source == "ma":
        if plot_magnitude:
            title += (
                f" Slope Magnitude "
                f"[|CFD MA-{moving_average_window} slope|]"
            )
        else:
            title += f" Slope [CFD MA-{moving_average_window} slope]"

    if plot_moving_average_of_slope and plot_slope_curve:
        title += f" [slope CFD MA window = {moving_average_window}]"

    if use_second_axis_for_moving_average_of_slope:
        title += " [moving average of slope on right axis]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 10: CUMULATIVE SLOPE
# ============================================================

def figure_cumulative_slope(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax_right = ax_left.twinx()

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            slope_x, raw_slope = compute_slope(x, y)
            cumulative_raw_slope = compute_cumulative_sum(raw_slope)

            normalized_cumulative_raw_slope = compute_cumsum_normalized_by_iteration(
                slope_x,
                cumulative_raw_slope,
            )

            x_plot, cumulative_raw_slope_plot = downsample_for_plotting(
                slope_x,
                cumulative_raw_slope,
                settings,
            )

            x_norm_plot, normalized_cumulative_raw_slope_plot = downsample_for_plotting(
                slope_x,
                normalized_cumulative_raw_slope,
                settings,
            )

            ax_left.plot(
                x_plot,
                cumulative_raw_slope_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.8,
                alpha=0.85,
                label=f"{dataset.filename.stem} cumulative raw slope",
                zorder=3,
            )

            ax_right.plot(
                x_norm_plot,
                normalized_cumulative_raw_slope_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.2,
                linestyle="-.",
                alpha=1.0,
                label=f"{dataset.filename.stem} cumulative raw slope / iteration",
                zorder=6,
            )

            if (
                settings.plot_moving_average_of_cumulative_sum
                and moving_average_window is not None
            ):
                x_avg, cumulative_avg = compute_moving_average(
                    slope_x,
                    cumulative_raw_slope,
                    moving_average_window,
                )

                x_avg_plot, cumulative_avg_plot = downsample_for_plotting(
                    x_avg,
                    cumulative_avg,
                    settings,
                )

                ax_left.plot(
                    x_avg_plot,
                    cumulative_avg_plot,
                    color=get_plot_color(dataset_index, settings, "tertiary"),
                    linewidth=3.0,
                    alpha=1.0,
                    label=(
                        f"{dataset.filename.stem} cumulative raw slope "
                        f"CFD MA-{moving_average_window}"
                    ),
                    zorder=7,
                )

        ax_left.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        ax_right.axhline(
            0.0,
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            zorder=1,
        )

        ax_left.set_ylabel(
            f"cumsum d({variable_label})/dIteration",
            fontsize=12,
        )

        ax_right.set_ylabel(
            f"cumsum d({variable_label})/dIteration / iteration",
            fontsize=12,
        )

        if settings.cumulative_slope_logy:
            ax_left.set_yscale("log")
            ax_right.set_yscale("log")

        ax_left.set_axisbelow(True)

        if settings.show_grid:
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

        merge_twin_axis_legends(
            ax_left,
            ax_right,
            fontsize=9,
        )

        apply_limits_if_enabled(
            ax_left,
            settings=settings,
            logy=settings.cumulative_slope_logy,
            keep_zero_visible=not settings.cumulative_slope_logy,
        )

        apply_limits_if_enabled(
            ax_right,
            settings=settings,
            logy=settings.cumulative_slope_logy,
            keep_zero_visible=not settings.cumulative_slope_logy,
        )

    title = f"{group.title} Cumulative Sum of Raw Data Slope"

    if settings.plot_moving_average_of_cumulative_sum:
        title += f" [cumsum CFD MA window = {moving_average_window}]"

    title += " [normalized cumsum on right axis]"

    finalize_figure(fig, axes, title)

    return fig

# ============================================================
# FIGURE 9: SLOPE OF MOVING RMS AVERAGE
# ============================================================

def figure_slope_of_rms_average(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    use_second_axis_for_moving_average_of_rms_slope = (
        settings.plot_moving_average_of_slope
        and moving_average_window is not None
    )

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        if use_second_axis_for_moving_average_of_rms_slope:
            ax_right = ax_left.twinx()
        else:
            ax_right = None

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_rms, y_rms = compute_moving_rms(
                x,
                y,
                moving_average_window,
            )

            slope_x, slope_y = compute_slope(
                x_rms,
                y_rms,
            )

            x_slope_plot, slope_y_plot = downsample_for_plotting(
                slope_x,
                slope_y,
                settings,
            )

            ax_left.plot(
                x_slope_plot,
                slope_y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.2,
                alpha=0.55,
                label=(
                    f"{dataset.filename.stem} "
                    f"RMS-{moving_average_window} slope"
                ),
                zorder=3,
            )

            if settings.plot_moving_average_of_slope:
                x_slope_avg, slope_avg = compute_moving_average_ignore_nan(
                    slope_x,
                    slope_y,
                    moving_average_window,
                )

                x_slope_avg_plot, slope_avg_plot = downsample_for_plotting(
                    x_slope_avg,
                    slope_avg,
                    settings,
                )

                if ax_right is not None:
                    slope_average_axis = ax_right
                    slope_average_zorder = 7
                else:
                    slope_average_axis = ax_left
                    slope_average_zorder = 6

                slope_average_axis.plot(
                    x_slope_avg_plot,
                    slope_avg_plot,
                    color=get_plot_color(dataset_index, settings, "secondary"),
                    linewidth=3.0,
                    alpha=1.0,
                    linestyle="-",
                    label=(
                        f"{dataset.filename.stem} moving avg of "
                        f"RMS-{moving_average_window} slope "
                        f"CFD MA-{moving_average_window}"
                    ),
                    zorder=slope_average_zorder,
                )

        ax_left.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        ylabel_left = f"d(RMS-{moving_average_window} {variable_label})/dIteration"
        ylabel_right = (
            f"moving avg of d(RMS-{moving_average_window})/dIteration "
            f"({variable_label})"
        )

        if ax_right is None:
            format_axis(
                ax_left,
                ylabel=ylabel_left,
                settings=settings,
                logy=settings.slope_logy,
            )

            apply_limits_if_enabled(
                ax_left,
                settings=settings,
                logy=settings.slope_logy,
                keep_zero_visible=not settings.slope_logy,
            )

        else:
            ax_left.set_ylabel(ylabel_left, fontsize=12)
            ax_right.set_ylabel(ylabel_right, fontsize=12)

            if settings.slope_logy:
                ax_left.set_yscale("log")
                ax_right.set_yscale("log")

            ax_left.set_axisbelow(True)

            if settings.show_grid:
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

            merge_twin_axis_legends(
                ax_left,
                ax_right,
                fontsize=9,
            )

            apply_limits_if_enabled(
                ax_left,
                settings=settings,
                logy=settings.slope_logy,
                keep_zero_visible=not settings.slope_logy,
            )

            apply_limits_if_enabled(
                ax_right,
                settings=settings,
                logy=settings.slope_logy,
                keep_zero_visible=not settings.slope_logy,
            )

    title = (
        f"{group.title} Slope of Moving RMS Average "
        f"[RMS-{moving_average_window} slope]"
    )

    if settings.plot_moving_average_of_slope:
        title += f" [slope CFD MA window = {moving_average_window}]"

    if use_second_axis_for_moving_average_of_rms_slope:
        title += " [moving average of slope on right axis]"

    finalize_figure(fig, axes, title)

    return fig


def figure_cumulative_slope_of_moving_average(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax_left, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax_right = ax_left.twinx()

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_avg, y_avg = compute_moving_average(
                x,
                y,
                moving_average_window,
            )

            slope_x, moving_average_slope = compute_slope(
                x_avg,
                y_avg,
            )

            cumulative_moving_average_slope = compute_cumulative_sum(
                moving_average_slope
            )

            normalized_cumulative_moving_average_slope = (
                compute_cumsum_normalized_by_iteration(
                    slope_x,
                    cumulative_moving_average_slope,
                )
            )

            x_plot, cumulative_moving_average_slope_plot = downsample_for_plotting(
                slope_x,
                cumulative_moving_average_slope,
                settings,
            )

            x_norm_plot, normalized_cumulative_moving_average_slope_plot = (
                downsample_for_plotting(
                    slope_x,
                    normalized_cumulative_moving_average_slope,
                    settings,
                )
            )

            ax_left.plot(
                x_plot,
                cumulative_moving_average_slope_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.8,
                alpha=0.85,
                label=(
                    f"{dataset.filename.stem} cumulative "
                    f"CFD MA-{moving_average_window} slope"
                ),
                zorder=3,
            )

            ax_right.plot(
                x_norm_plot,
                normalized_cumulative_moving_average_slope_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.2,
                linestyle="-.",
                alpha=1.0,
                label=(
                    f"{dataset.filename.stem} cumulative "
                    f"CFD MA-{moving_average_window} slope / iteration"
                ),
                zorder=6,
            )

        ax_left.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

        ax_right.axhline(
            0.0,
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.5,
            zorder=1,
        )

        ax_left.set_ylabel(
            f"cumsum d(CFD MA-{moving_average_window} {variable_label})/dIteration",
            fontsize=12,
        )

        ax_right.set_ylabel(
            (
                f"cumsum d(CFD MA-{moving_average_window} {variable_label})"
                f"/dIteration / iteration"
            ),
            fontsize=12,
        )

        if settings.cumulative_slope_logy:
            ax_left.set_yscale("log")
            ax_right.set_yscale("log")

        ax_left.set_axisbelow(True)

        if settings.show_grid:
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

        merge_twin_axis_legends(
            ax_left,
            ax_right,
            fontsize=9,
        )

        apply_limits_if_enabled(
            ax_left,
            settings=settings,
            logy=settings.cumulative_slope_logy,
            keep_zero_visible=not settings.cumulative_slope_logy,
        )

        apply_limits_if_enabled(
            ax_right,
            settings=settings,
            logy=settings.cumulative_slope_logy,
            keep_zero_visible=not settings.cumulative_slope_logy,
        )

    title = (
        f"{group.title} Cumulative Sum of Moving-Average Slope "
        f"[CFD MA-{moving_average_window} slope] "
        f"[normalized cumsum on right axis]"
    )

    finalize_figure(fig, axes, title)

    return fig


def figure_raw_slope_sign_indicator(
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
):
    loaded_group = load_group_data(group, settings)

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            slope_x, raw_slope = compute_slope(x, y)

            slope_sign_for_plot = compute_slope_sign_indicator(
                raw_slope,
                use_nan_for_zero_or_invalid=True,
            )

            x_sign_plot, sign_plot = downsample_for_plotting(
                slope_x,
                slope_sign_for_plot,
                settings,
            )

            ax.plot(
                x_sign_plot,
                sign_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.1,
                alpha=0.55,
                drawstyle="steps-post",
                label=(
                    f"{dataset.filename.stem} raw slope sign "
                    "[+1 positive, -1 negative]"
                ),
                zorder=3,
            )

            if (
                settings.plot_moving_average_of_raw_slope_sign_indicator
                and moving_average_window is not None
            ):
                x_sign_avg, sign_avg = compute_moving_average_ignore_nan(
                    slope_x,
                    slope_sign_for_plot,
                    moving_average_window,
                )

                x_sign_avg_plot, sign_avg_plot = downsample_for_plotting(
                    x_sign_avg,
                    sign_avg,
                    settings,
                )

                ax.plot(
                    x_sign_avg_plot,
                    sign_avg_plot,
                    color=get_plot_color(dataset_index, settings, "secondary"),
                    linewidth=3.0,
                    alpha=1.0,
                    label=(
                        f"{dataset.filename.stem} moving avg of raw slope sign "
                        f"CFD MA-{moving_average_window}"
                    ),
                    zorder=6,
                )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

        ax.axhline(
            1.0,
            color="gray",
            linewidth=0.8,
            linestyle=":",
            alpha=0.45,
            zorder=1,
        )

        ax.axhline(
            -1.0,
            color="gray",
            linewidth=0.8,
            linestyle=":",
            alpha=0.45,
            zorder=1,
        )

        format_axis(
            ax,
            ylabel=f"sign d({variable_label})/dIteration",
            settings=settings,
            logy=False,
        )

        ax.set_ylim(-1.15, 1.15)

    title = (
        f"{group.title} Raw Slope Sign Indicator "
        "[+1 positive slope, -1 negative slope]"
    )

    if (
        settings.plot_moving_average_of_raw_slope_sign_indicator
        and moving_average_window is not None
    ):
        title += f" [sign CFD MA window = {moving_average_window}]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# PDF / OUTPUT HELPERS
# ============================================================

def format_iteration_value_for_filename(value: float | None) -> str | None:
    if value is None:
        return None

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "p")


def get_iteration_range_suffix(settings: Settings) -> str:
    if (
        settings.min_iteration_to_include is None
        and settings.max_iteration_to_include is None
    ):
        return ""

    start_text = format_iteration_value_for_filename(
        settings.min_iteration_to_include
    )

    end_text = format_iteration_value_for_filename(
        settings.max_iteration_to_include
    )

    if start_text is None:
        start_text = "start"

    if end_text is None:
        end_text = "end"

    return f"_iter_{start_text}_to_{end_text}"


def make_pdf_name(
    base_name: str | Path,
    settings: Settings,
    moving_average_window: int | None = None,
) -> Path:
    base_name = Path(base_name)

    suffix_parts = []

    iteration_range_suffix = get_iteration_range_suffix(settings)

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


def add_enabled_figures_for_group(
    pdf: PdfPages,
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
    plot_moving_average: bool,
) -> None:
    print(f"Adding group: {group.title}")

    fig = figure_main_data(
        group=group,
        settings=settings,
        moving_average_window=moving_average_window,
        plot_moving_average=plot_moving_average,
    )
    add_page_to_pdf(pdf, fig, settings)

    if not plot_moving_average:
        return

    if moving_average_window is None:
        return

    if settings.plot_online_cumulative_sum_of_moving_average_comparison:
        fig = figure_online_cumulative_ma_comparison(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_of_cumulative_sum_of_moving_average_comparison:
        fig = figure_slope_of_cumulative_ma_comparison(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_raw_ma_and_cumulative_sum_second_axis:
        fig = figure_raw_ma_cumsum_second_axis(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_moving_average_ratio:
        fig = figure_moving_average_ratio(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_rms_ratio:
        fig = figure_rms_ratio(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_lag1_autocorrelation:
        fig = figure_lag1_autocorrelation(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_short_long_rms:
        fig = figure_short_long_rms(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if not settings.calculate_slope:
        return

    if settings.plot_slope_of_raw:
        fig = figure_slope(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
            slope_source="raw",
            plot_magnitude=False,
            plot_positive_only=False,
            plot_slope_curve=True,
            plot_moving_average_of_slope=settings.plot_moving_average_of_slope,
            slope_logy=settings.slope_logy,
        )
        add_page_to_pdf(pdf, fig, settings)

        if settings.plot_cumulative_sum_of_error_slope:
            fig = figure_cumulative_slope(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
            )
            add_page_to_pdf(pdf, fig, settings)

        if (
            settings.plot_raw_slope_moving_average_only
            and settings.plot_moving_average_of_slope
        ):
            fig = figure_slope(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
                slope_source="raw",
                plot_magnitude=False,
                plot_positive_only=False,
                plot_slope_curve=False,
                plot_moving_average_of_slope=True,
                slope_logy=settings.slope_logy,
            )
            add_page_to_pdf(pdf, fig, settings)

        if settings.plot_positive_raw_slopes_only:
            fig = figure_slope(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
                slope_source="raw",
                plot_magnitude=False,
                plot_positive_only=True,
                plot_slope_curve=True,
                plot_moving_average_of_slope=settings.plot_moving_average_of_slope,
                slope_logy=settings.slope_logy,
            )
            add_page_to_pdf(pdf, fig, settings)

        if settings.plot_slope_magnitude:
            fig = figure_slope(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
                slope_source="raw",
                plot_magnitude=True,
                plot_positive_only=False,
                plot_slope_curve=True,
                plot_moving_average_of_slope=settings.plot_moving_average_of_slope,
                slope_logy=settings.slope_magnitude_logy,
            )
            add_page_to_pdf(pdf, fig, settings)


        if settings.plot_raw_slope_sign_running_total:
            fig = figure_raw_slope_sign_running_total(
                group=group,
                settings=settings,
            )
            add_page_to_pdf(pdf, fig, settings)

        if settings.plot_raw_slope_sign_indicator:
            fig = figure_raw_slope_sign_indicator(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
            )
            add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_of_moving_average:
        fig = figure_slope(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
            slope_source="ma",
            plot_magnitude=False,
            plot_positive_only=False,
            plot_slope_curve=True,
            plot_moving_average_of_slope=settings.plot_moving_average_of_slope,
            slope_logy=settings.slope_logy,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_cumulative_sum_of_moving_average_slope:
        fig = figure_cumulative_slope_of_moving_average(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_of_rms_average:
        fig = figure_slope_of_rms_average(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)


        if settings.plot_slope_magnitude:
            fig = figure_slope(
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
                slope_source="ma",
                plot_magnitude=True,
                plot_positive_only=False,
                plot_slope_curve=True,
                plot_moving_average_of_slope=settings.plot_moving_average_of_slope,
                slope_logy=settings.slope_magnitude_logy,
            )
            add_page_to_pdf(pdf, fig, settings)


def write_pdf(
    pdf_path: Path,
    settings: Settings,
    figure_groups: list[FigureGroup],
    moving_average_window: int | None,
    plot_moving_average: bool,
) -> None:
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
        for group in figure_groups:
            add_enabled_figures_for_group(
                pdf=pdf,
                group=group,
                settings=settings,
                moving_average_window=moving_average_window,
                plot_moving_average=plot_moving_average,
            )

    print(f"Saved PDF to: {pdf_path}")


# ============================================================
# MAIN
# ============================================================

def print_run_summary(settings: Settings) -> None:
    if settings.min_iteration_to_include is None:
        print("Minimum iteration: using first available iteration")
    else:
        print(f"Minimum iteration: {settings.min_iteration_to_include}")

    if settings.max_iteration_to_include is None:
        print("Maximum iteration: using final available iteration")
    else:
        print(f"Maximum iteration: {settings.max_iteration_to_include}")

    print(f"Output directory: {settings.output_dir}")
    print("Existing files in the output directory will not be deleted.")

def normalize_pdf_output_name(output_base_name: str) -> str:
    output_base_name = output_base_name.strip()

    if output_base_name == "":
        raise ValueError("Output base name cannot be empty.")

    output_path = Path(output_base_name)

    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_name(f"{output_path.name}.pdf")

    return str(output_path)


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate comparison PDF figures from the configured data files."
        )
    )

    parser.add_argument(
        "output_base_name",
        help=(
            "Base name for the output PDF. The .pdf extension is optional. "
            "For example: run_01 or run_01.pdf"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_command_line_arguments()

    pdf_output_file = normalize_pdf_output_name(args.output_base_name)

    settings = replace(
        SETTINGS,
        pdf_output_file=pdf_output_file,
    )

    figure_groups = FIGURE_GROUPS

    print_run_summary(settings)

    if settings.plot_moving_average:
        for moving_average_window in settings.moving_average_windows:
            pdf_name = make_pdf_name(
                settings.pdf_output_file,
                settings=settings,
                moving_average_window=moving_average_window,
            )

            pdf_path = settings.output_dir / pdf_name

            write_pdf(
                pdf_path=pdf_path,
                settings=settings,
                figure_groups=figure_groups,
                moving_average_window=moving_average_window,
                plot_moving_average=True,
            )

    else:
        pdf_name = make_pdf_name(
            settings.pdf_output_file,
            settings=settings,
            moving_average_window=None,
        )

        pdf_path = settings.output_dir / pdf_name

        write_pdf(
            pdf_path=pdf_path,
            settings=settings,
            figure_groups=figure_groups,
            moving_average_window=None,
            plot_moving_average=False,
        )

    if settings.show_figures:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()