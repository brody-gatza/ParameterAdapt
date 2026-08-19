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
    plot_every_nth_point: int | None = None

    # ----------------------------
    # Moving/running averages
    # ----------------------------
    plot_moving_average: bool = True
    moving_average_windows: list[int] = field(default_factory=lambda: [1000])

    plot_running_average: bool = True

    # ----------------------------
    # Sampling-frequency figure
    # ----------------------------
    sampling_frequency_filename: str = "sampling_freq.txt"
    plot_sampling_frequency: bool = True

    # ----------------------------
    # Slope-counter figure
    # ----------------------------
    plot_slope_counter_data: bool = False
    slope_counter_suffix: str = "_slope_counter"

    # ----------------------------
    # Slope-ratio figure
    # ----------------------------
    plot_slope_ratio_data: bool = False
    slope_ratio_suffix: str = "_slope_ratio"

    # ----------------------------
    # Slope-counter / slope-ratio overlay figure
    # ----------------------------
    plot_slope_counter_ratio_overlay_data: bool = False

    # ----------------------------
    # Saved moving-average figures
    # ----------------------------
    plot_saved_moving_average_overlay_data: bool = True
    plot_moving_average_counter_data: bool = True
    plot_moving_average_counter_sum_data: bool = True

    short_ma_suffix: str = "_short_ma"
    long_ma_suffix: str = "_long_ma"
    ma_counter_suffix: str = "_ma_counter"
    ma_counter_sum_suffix: str = "_ma_counter_sum"

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
        title="Primitive Variables - Avg",
        filenames=[
            "prim_interp_avg.txt",
            # "prim_proj_avg.txt",
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

    # Guard against a single-row file collapsing to 1D.
    if data.ndim == 1:
        data = data.reshape(1, -1)

    iterations = data[:, 0]
    variables = data[:, 1:]

    iterations, variables = apply_iteration_limits(
        iterations,
        variables,
        filename,
        settings,
    )

    return LoadedDataset(
        filename=filename,
        iterations=iterations,
        variables=variables,
    )


def load_group_data(
    group: FigureGroup,
    settings: Settings,
    filenames_override: list[str] | None = None,
) -> LoadedGroup:
    filenames = (
        group.filenames if filenames_override is None else filenames_override
    )

    datasets: list[LoadedDataset] = []
    max_num_vars = 0

    for filename_text in filenames:
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


def get_slope_counter_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.slope_counter_suffix}{filename.suffix}"
    )


def get_slope_ratio_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.slope_ratio_suffix}{filename.suffix}"
    )


def get_short_ma_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.short_ma_suffix}{filename.suffix}"
    )


def get_long_ma_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.long_ma_suffix}{filename.suffix}"
    )


def get_ma_counter_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.ma_counter_suffix}{filename.suffix}"
    )


def get_ma_counter_sum_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.ma_counter_sum_suffix}{filename.suffix}"
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


def compute_running_average(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)

    cumulative_sum = np.cumsum(y)
    sample_count = np.arange(1, len(y) + 1)
    running_average = cumulative_sum / sample_count

    return x, running_average


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


# ============================================================
# FIGURE 1: MAIN DATA (raw + running average + moving average)
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
# FIGURE: RAW DATA + SAVED SHORT/LONG MOVING AVERAGES
# ============================================================

def figure_saved_moving_average_overlay_data(
    group: FigureGroup,
    settings: Settings,
):
    raw_group = load_group_data(group, settings)

    short_ma_filenames = [
        str(get_short_ma_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    long_ma_filenames = [
        str(get_long_ma_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    short_ma_group = load_group_data(
        group,
        settings,
        filenames_override=short_ma_filenames,
    )

    long_ma_group = load_group_data(
        group,
        settings,
        filenames_override=long_ma_filenames,
    )

    fig, axes = create_figure(raw_group.variable_indices)

    for ax, var_index in zip(axes, raw_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        for dataset_index, raw_dataset in enumerate(raw_group.datasets):
            short_ma_dataset = short_ma_group.datasets[dataset_index]
            long_ma_dataset = long_ma_group.datasets[dataset_index]

            x_raw = raw_dataset.iterations
            y_raw = raw_dataset.variables[:, var_index]

            x_short = short_ma_dataset.iterations
            y_short = short_ma_dataset.variables[:, var_index]

            x_long = long_ma_dataset.iterations
            y_long = long_ma_dataset.variables[:, var_index]

            x_raw_plot, y_raw_plot = downsample_for_plotting(
                x_raw,
                y_raw,
                settings,
            )

            x_short_plot, y_short_plot = downsample_for_plotting(
                x_short,
                y_short,
                settings,
            )

            x_long_plot, y_long_plot = downsample_for_plotting(
                x_long,
                y_long,
                settings,
            )

            ax.plot(
                x_raw_plot,
                y_raw_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.0,
                alpha=0.55,
                label=f"{raw_dataset.filename.stem} raw",
                zorder=3,
            )

            ax.plot(
                x_short_plot,
                y_short_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=2.4,
                alpha=0.95,
                linestyle="-",
                label=f"{raw_dataset.filename.stem} short MA",
                zorder=5,
            )

            ax.plot(
                x_long_plot,
                y_long_plot,
                color=get_plot_color(dataset_index, settings, "tertiary"),
                linewidth=2.4,
                alpha=0.95,
                linestyle="--",
                label=f"{raw_dataset.filename.stem} long MA",
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

    title = f"{group.title} Raw Data with Saved Moving Averages"

    if group.logy:
        title += " [log-y]"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE: MOVING-AVERAGE COUNTER
# ============================================================

def figure_moving_average_counter_data(
    group: FigureGroup,
    settings: Settings,
):
    ma_counter_filenames = [
        str(get_ma_counter_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    loaded_group = load_group_data(
        group,
        settings,
        filenames_override=ma_counter_filenames,
    )

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
            label="counter = 0",
            zorder=2,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_plot, y_plot = downsample_for_plotting(x, y, settings)

            ax.plot(
                x_plot,
                y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.6,
                alpha=0.9,
                label=f"{dataset.filename.stem}",
                zorder=3,
            )

        format_axis(
            ax,
            ylabel=f"MA counter ({variable_label})",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    title = f"{group.title} Moving-Average Counter"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE: MOVING-AVERAGE COUNTER SUM
# ============================================================

def figure_moving_average_counter_sum_data(
    group: FigureGroup,
    settings: Settings,
):
    ma_counter_sum_filenames = [
        str(get_ma_counter_sum_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    loaded_group = load_group_data(
        group,
        settings,
        filenames_override=ma_counter_sum_filenames,
    )

    fig, axes = create_figure([0])
    ax = axes[0]

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
        label="sum = 0",
        zorder=2,
    )

    for dataset_index, dataset in enumerate(loaded_group.datasets):
        x = dataset.iterations
        y = dataset.variables[:, 0]

        x_plot, y_plot = downsample_for_plotting(
            x,
            y,
            settings,
        )

        ax.plot(
            x_plot,
            y_plot,
            color=get_plot_color(dataset_index, settings, "primary"),
            linewidth=1.8,
            alpha=0.9,
            label=f"{dataset.filename.stem}",
            zorder=3,
        )

    format_axis(
        ax,
        ylabel="MA counter sum",
        settings=settings,
        logy=False,
    )

    apply_limits_if_enabled(
        ax,
        settings=settings,
        logy=False,
        keep_zero_visible=True,
    )

    title = f"{group.title} Moving-Average Counter Sum"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 2: SLOPE-COUNTER RAW DATA
# (same variable layout as the group, loaded from
#  "<original_filename>_slope_counter.txt")
# ============================================================

def figure_slope_counter_data(
    group: FigureGroup,
    settings: Settings,
):
    slope_counter_filenames = [
        str(get_slope_counter_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    loaded_group = load_group_data(
        group,
        settings,
        filenames_override=slope_counter_filenames,
    )

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

            x_plot, y_plot = downsample_for_plotting(x, y, settings)

            ax.plot(
                x_plot,
                y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.6,
                alpha=0.9,
                label=f"{dataset.filename.stem}",
                zorder=3,
            )

        format_axis(
            ax,
            ylabel=f"slope counter ({variable_label})",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=False,
        )

    title = f"{group.title} Slope Counter Raw Data"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 3: SLOPE-RATIO RAW DATA
# (same variable layout as the group, loaded from
#  "<original_filename>_slope_ratio.txt")
# ============================================================

def figure_slope_ratio_data(
    group: FigureGroup,
    settings: Settings,
):
    slope_ratio_filenames = [
        str(get_slope_ratio_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    loaded_group = load_group_data(
        group,
        settings,
        filenames_override=slope_ratio_filenames,
    )

    fig, axes = create_figure(loaded_group.variable_indices)

    for ax, var_index in zip(axes, loaded_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
            label="ratio = 1",
            zorder=2,
        )

        for dataset_index, dataset in enumerate(loaded_group.datasets):
            x = dataset.iterations
            y = dataset.variables[:, var_index]

            x_plot, y_plot = downsample_for_plotting(x, y, settings)

            ax.plot(
                x_plot,
                y_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.6,
                alpha=0.9,
                label=f"{dataset.filename.stem}",
                zorder=3,
            )

        format_axis(
            ax,
            ylabel=f"slope ratio ({variable_label})",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=False,
        )

    title = f"{group.title} Slope Ratio Raw Data"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 4: SLOPE-COUNTER / SLOPE-RATIO OVERLAY DATA
# (same variable layout as the group; slope counter on left axis,
#  slope ratio on right axis)
# ============================================================

def figure_slope_counter_ratio_overlay_data(
    group: FigureGroup,
    settings: Settings,
):
    slope_counter_filenames = [
        str(get_slope_counter_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    slope_ratio_filenames = [
        str(get_slope_ratio_filename(Path(filename_text), settings))
        for filename_text in group.filenames
    ]

    counter_group = load_group_data(
        group,
        settings,
        filenames_override=slope_counter_filenames,
    )

    ratio_group = load_group_data(
        group,
        settings,
        filenames_override=slope_ratio_filenames,
    )

    fig, axes = create_figure(counter_group.variable_indices)

    for ax_counter, var_index in zip(axes, counter_group.variable_indices):
        variable_label = get_variable_label_for_group(
            group=group,
            settings=settings,
            var_index=var_index,
        )

        ax_ratio = ax_counter.twinx()

        for dataset_index, counter_dataset in enumerate(counter_group.datasets):
            ratio_dataset = ratio_group.datasets[dataset_index]

            x_counter = counter_dataset.iterations
            y_counter = counter_dataset.variables[:, var_index]

            x_ratio = ratio_dataset.iterations
            y_ratio = ratio_dataset.variables[:, var_index]

            x_counter_plot, y_counter_plot = downsample_for_plotting(
                x_counter,
                y_counter,
                settings,
            )

            x_ratio_plot, y_ratio_plot = downsample_for_plotting(
                x_ratio,
                y_ratio,
                settings,
            )

            ax_counter.plot(
                x_counter_plot,
                y_counter_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=1.6,
                alpha=0.9,
                label=f"{counter_dataset.filename.stem}",
                zorder=3,
            )

            ax_ratio.plot(
                x_ratio_plot,
                y_ratio_plot,
                color=get_plot_color(dataset_index, settings, "secondary"),
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                label=f"{ratio_dataset.filename.stem}",
                zorder=4,
            )

        ax_ratio.axhline(
            1.0,
            color="black",
            linewidth=1.0,
            linestyle=":",
            alpha=0.65,
            label="ratio = 1",
            zorder=2,
        )

        ax_counter.set_ylabel(
            f"slope counter ({variable_label})",
            fontsize=12,
        )

        ax_ratio.set_ylabel(
            f"slope ratio ({variable_label})",
            fontsize=12,
        )

        ax_counter.set_axisbelow(True)

        if settings.show_grid:
            ax_counter.grid(
                True,
                which="both",
                linestyle="--",
                linewidth=0.7,
                alpha=0.6,
                zorder=0,
            )

        counter_handles, counter_labels = ax_counter.get_legend_handles_labels()
        ratio_handles, ratio_labels = ax_ratio.get_legend_handles_labels()

        ax_counter.legend(
            counter_handles + ratio_handles,
            counter_labels + ratio_labels,
            fontsize=10,
            frameon=True,
        )

        ax_counter.tick_params(axis="both", labelsize=11)
        ax_ratio.tick_params(axis="both", labelsize=11)

        apply_limits_if_enabled(
            ax_counter,
            settings=settings,
            logy=False,
            keep_zero_visible=False,
        )

        apply_limits_if_enabled(
            ax_ratio,
            settings=settings,
            logy=False,
            keep_zero_visible=False,
        )

    title = f"{group.title} Slope Counter and Slope Ratio Overlay"

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 5: SAMPLING FREQUENCY (raw + moving average)
# ============================================================

def figure_sampling_frequency(
    settings: Settings,
):
    filename = Path(settings.sampling_frequency_filename)

    dataset = load_dataset(filename=filename, settings=settings)

    fig, axes = create_figure([0])
    ax = axes[0]

    x = dataset.iterations
    y = dataset.variables[:, 0]

    x_plot, y_plot = downsample_for_plotting(x, y, settings)

    ax.plot(
        x_plot,
        y_plot,
        color=get_plot_color(0, settings, "primary"),
        linewidth=1.0,
        alpha=0.70,
        label="sampling frequency",
        zorder=3,
    )

    if settings.plot_running_average:
        x_avg, y_avg = compute_running_average(x, y)

        x_avg_plot, y_avg_plot = downsample_for_plotting(
            x_avg,
            y_avg,
            settings,
        )

        ax.plot(
            x_avg_plot,
            y_avg_plot,
            color=get_plot_color(0, settings, "secondary"),
            linewidth=3.0,
            alpha=1.0,
            label="sampling frequency running avg",
            zorder=5,
        )

    format_axis(
        ax,
        ylabel="Sampling Frequency",
        settings=settings,
        logy=False,
    )

    title = "Sampling Frequency"

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

    return base_name.with_name(
        f"{base_name.stem}_counter{base_name.suffix}"
    )


def add_enabled_figures_for_group(
    pdf: PdfPages,
    group: FigureGroup,
    settings: Settings,
    moving_average_window: int | None,
    plot_moving_average: bool,
) -> None:
    print(f"Adding group: {group.title}")

    if settings.plot_saved_moving_average_overlay_data:
        fig = figure_saved_moving_average_overlay_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_moving_average_counter_data:
        fig = figure_moving_average_counter_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_moving_average_counter_sum_data:
        fig = figure_moving_average_counter_sum_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    fig = figure_main_data(
        group=group,
        settings=settings,
        moving_average_window=moving_average_window,
        plot_moving_average=plot_moving_average,
    )
    add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_counter_data:
        fig = figure_slope_counter_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_counter_ratio_overlay_data:
        fig = figure_slope_counter_ratio_overlay_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    if settings.plot_slope_ratio_data:
        fig = figure_slope_ratio_data(
            group=group,
            settings=settings,
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
        # ------------------------------------------------------------
        # PAGE 1: Sampling frequency
        # ------------------------------------------------------------
        if settings.plot_sampling_frequency:
            print("Adding sampling frequency figure as page 1")
            fig = figure_sampling_frequency(
                settings=settings,
            )
            add_page_to_pdf(pdf, fig, settings)

        # ------------------------------------------------------------
        # Remaining pages: Primitive groups, then conservative groups
        # according to FIGURE_GROUPS order
        # ------------------------------------------------------------
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
            "Generate lightweight comparison PDF figures from the "
            "configured data files."
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
