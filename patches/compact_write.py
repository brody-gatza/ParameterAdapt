#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


SETTINGS_INSERT = '''
    # ----------------------------
    # Rolling normalized trend score
    # ----------------------------
    plot_rolling_normalized_trend_score: bool = True

    # Window sizes, in samples, for detecting upward/downward trends.
    # The plotted metric is:
    #
    #     normalized trend score = slope over window / std over window
    #
    # Time is normalized from 0 to 1 inside each window, so this is roughly:
    #
    #     standard deviations of increase across the window
    rolling_normalized_trend_windows: list[int] = field(
        default_factory=lambda: [500]
    )

    # If True, compute the metric from the moving-averaged data when available.
    # This is usually better for noisy CFD/convergence-type signals.
    rolling_normalized_trend_use_moving_average: bool = True

    # If the local standard deviation is smaller than this value,
    # the trend score is set to NaN.
    rolling_normalized_trend_min_std: float = 1.0e-30

    # Visual reference thresholds. Positive values indicate upward trends.
    rolling_normalized_trend_thresholds: list[float] = field(
        default_factory=lambda: [0.25, 0.50, 1.00]
    )

    # If True, use Theil-Sen median pairwise slope in each window.
    # More robust to outliers, but much slower for large windows.
    # If False, use ordinary least-squares slope.
    rolling_normalized_trend_use_theil_sen: bool = False

'''


COMPUTE_INSERT = '''
def compute_theil_sen_slope_for_window(
    x_window: np.ndarray,
    y_window: np.ndarray,
) -> float:
    """
    Robust Theil-Sen slope estimate for one window.

    x_window should already be normalized if you want the slope to represent
    total change across one local window.
    """
    x_window = np.asarray(x_window, dtype=float)
    y_window = np.asarray(y_window, dtype=float)

    finite_mask = np.isfinite(x_window) & np.isfinite(y_window)
    x_window = x_window[finite_mask]
    y_window = y_window[finite_mask]

    n = len(y_window)

    if n < 2:
        return np.nan

    slopes = []

    for i in range(n - 1):
        dx = x_window[i + 1:] - x_window[i]
        dy = y_window[i + 1:] - y_window[i]

        valid_mask = dx != 0.0

        if np.any(valid_mask):
            slopes.extend((dy[valid_mask] / dx[valid_mask]).tolist())

    if len(slopes) == 0:
        return np.nan

    return float(np.median(slopes))


def compute_least_squares_slope_for_window(
    x_window: np.ndarray,
    y_window: np.ndarray,
) -> float:
    """
    Fast ordinary least-squares slope for one window.

    x_window should already be normalized if you want the slope to represent
    total change across one local window.
    """
    x_window = np.asarray(x_window, dtype=float)
    y_window = np.asarray(y_window, dtype=float)

    finite_mask = np.isfinite(x_window) & np.isfinite(y_window)
    x_window = x_window[finite_mask]
    y_window = y_window[finite_mask]

    if len(y_window) < 2:
        return np.nan

    x_centered = x_window - np.mean(x_window)
    y_centered = y_window - np.mean(y_window)

    denominator = np.sum(x_centered * x_centered)

    if denominator == 0.0:
        return np.nan

    return float(np.sum(x_centered * y_centered) / denominator)


def compute_rolling_normalized_trend_score(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    min_std: float,
    use_theil_sen: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a scale-independent rolling trend score.

    The metric is:

        normalized_trend_score = slope_over_window / std_over_window

    Time is normalized to [0, 1] inside each window, so the score is roughly:

        standard deviations of signal increase across the window

    Positive values indicate upward trend.
    Negative values indicate downward trend.

    Example:
        +0.25  weak upward trend
        +0.50  moderate upward trend
        +1.00  strong upward trend
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window <= 1:
        raise ValueError("Rolling normalized trend window must be greater than 1.")

    if min_std < 0.0:
        raise ValueError("rolling_normalized_trend_min_std must be nonnegative.")

    trend_score = np.full(len(y), np.nan)

    for i in range(len(y)):
        window_start = max(0, i - window + 1)

        x_window = x[window_start : i + 1]
        y_window = y[window_start : i + 1]

        finite_mask = np.isfinite(x_window) & np.isfinite(y_window)
        x_window = x_window[finite_mask]
        y_window = y_window[finite_mask]

        if len(y_window) < 3:
            continue

        y_std = np.std(y_window, ddof=0)

        if not np.isfinite(y_std):
            continue

        if y_std <= min_std:
            continue

        x_min = np.min(x_window)
        x_max = np.max(x_window)

        if not np.isfinite(x_min):
            continue

        if not np.isfinite(x_max):
            continue

        if x_max == x_min:
            continue

        # Normalize local time from 0 to 1.
        # This makes the slope represent total signal change across this window.
        x_normalized = (x_window - x_min) / (x_max - x_min)

        if use_theil_sen:
            slope = compute_theil_sen_slope_for_window(
                x_normalized,
                y_window,
            )
        else:
            slope = compute_least_squares_slope_for_window(
                x_normalized,
                y_window,
            )

        if not np.isfinite(slope):
            continue

        trend_score[i] = slope / y_std

    return x, trend_score


'''


FIGURE_INSERT = '''
# ============================================================
# FIGURE 1C: ROLLING NORMALIZED TREND SCORE
# ============================================================

def figure_rolling_normalized_trend_score(
    group: FigureGroup,
    settings: Settings,
    rolling_normalized_trend_window: int,
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

            if (
                settings.rolling_normalized_trend_use_moving_average
                and moving_average_window is not None
            ):
                x_source, y_source = compute_moving_average(
                    x,
                    y,
                    moving_average_window,
                )
                source_label = f"CFD MA-{moving_average_window}"
            else:
                x_source = x
                y_source = y
                source_label = "raw"

            x_trend, trend_score = compute_rolling_normalized_trend_score(
                x=x_source,
                y=y_source,
                window=rolling_normalized_trend_window,
                min_std=settings.rolling_normalized_trend_min_std,
                use_theil_sen=settings.rolling_normalized_trend_use_theil_sen,
            )

            x_trend_plot, trend_score_plot = downsample_for_plotting(
                x_trend,
                trend_score,
                settings,
            )

            ax.plot(
                x_trend_plot,
                trend_score_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.2,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} normalized trend "
                    f"{source_label}, window-{rolling_normalized_trend_window}"
                ),
                zorder=5,
            )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )

        for threshold in settings.rolling_normalized_trend_thresholds:
            if threshold <= 0.0:
                continue

            ax.axhline(
                threshold,
                color="gray",
                linewidth=0.9,
                linestyle=":",
                alpha=0.55,
                zorder=1,
            )

            ax.axhline(
                -threshold,
                color="gray",
                linewidth=0.9,
                linestyle=":",
                alpha=0.35,
                zorder=1,
            )

        format_axis(
            ax,
            ylabel=f"normalized trend score ({variable_label})",
            settings=settings,
            logy=False,
        )

        apply_limits_if_enabled(
            ax,
            settings=settings,
            logy=False,
            keep_zero_visible=True,
        )

    if settings.rolling_normalized_trend_use_theil_sen:
        slope_method_text = "Theil-Sen"
    else:
        slope_method_text = "least-squares"

    title = (
        f"{group.title} Rolling Normalized Trend Score "
        f"[window={rolling_normalized_trend_window}, "
        f"slope={slope_method_text}]"
    )

    if (
        settings.rolling_normalized_trend_use_moving_average
        and moving_average_window is not None
    ):
        title += f" [computed from CFD MA-{moving_average_window}]"
    else:
        title += " [computed from raw signal]"

    finalize_figure(fig, axes, title)

    return fig


'''


PDF_HOOK_INSERT = '''
    if settings.plot_rolling_normalized_trend_score:
        for rolling_trend_window in settings.rolling_normalized_trend_windows:
            print(
                "Adding rolling normalized trend score figure "
                f"for window = {rolling_trend_window}"
            )

            fig = figure_rolling_normalized_trend_score(
                group=group,
                settings=settings,
                rolling_normalized_trend_window=rolling_trend_window,
                moving_average_window=moving_average_window,
            )
            add_page_to_pdf(pdf, fig, settings)

'''


def normalize_newlines(text: str) -> str:
    return text.replace("\\r\\n", "\\n").replace("\\r", "\\n")


def insert_before_marker(
    text: str,
    marker: str,
    insert: str,
    description: str,
) -> str:
    if insert.strip() in text:
        print(f"[skip] {description} already present")
        return text

    index = text.find(marker)

    if index < 0:
        raise RuntimeError(
            f"Could not find marker for {description}:\\n"
            f"{marker!r}"
        )

    print(f"[add] {description}")
    return text[:index] + insert + text[index:]


def insert_after_marker(
    text: str,
    marker: str,
    insert: str,
    description: str,
) -> str:
    if insert.strip() in text:
        print(f"[skip] {description} already present")
        return text

    index = text.find(marker)

    if index < 0:
        raise RuntimeError(
            f"Could not find marker for {description}:\\n"
            f"{marker!r}"
        )

    index += len(marker)

    print(f"[add] {description}")
    return text[:index] + insert + text[index:]


def patch_text(text: str) -> str:
    text = normalize_newlines(text)

    text = insert_before_marker(
        text=text,
        marker="    # ----------------------------\n    # Main derived figure toggles",
        insert=SETTINGS_INSERT,
        description="Settings fields",
    )

    text = insert_before_marker(
        text=text,
        marker="def compute_cumulative_sum(y: np.ndarray) -> np.ndarray:",
        insert=COMPUTE_INSERT,
        description="trend score computation helpers",
    )

    text = insert_before_marker(
        text=text,
        marker="# ============================================================\n# FIGURE 2: CUMULATIVE MOVING-AVERAGE COMPARISON",
        insert=FIGURE_INSERT,
        description="trend score figure function",
    )

    text = insert_before_marker(
        text=text,
        marker="    if not plot_moving_average:\n        return",
        insert=PDF_HOOK_INSERT,
        description="PDF generation hook",
    )

    return text


def patch_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    original_text = path.read_text()
    patched_text = patch_text(original_text)

    if patched_text == normalize_newlines(original_text):
        print("[done] No changes made")
        return

    backup_path = path.with_suffix(path.suffix + ".bak")
    backup_path.write_text(original_text)
    path.write_text(patched_text)

    print(f"[done] Patched file: {path}")
    print(f"[done] Backup file:  {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a Python-based patch that adds a rolling normalized "
            "trend score figure to the plotting script."
        )
    )

    parser.add_argument(
        "script",
        type=Path,
        help="Path to the Python plotting script to patch.",
    )

    args = parser.parse_args()

    patch_file(args.script)


if __name__ == "__main__":
    main()