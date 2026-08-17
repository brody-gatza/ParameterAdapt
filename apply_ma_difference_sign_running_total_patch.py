#!/usr/bin/env python3
from pathlib import Path

TARGET_FILE = Path("plot_errors.py")

text = TARGET_FILE.read_text()

# ----------------------------------------------------------------------
# 1. Add new Settings toggle
# ----------------------------------------------------------------------
old = """    plot_online_cumulative_sum_of_moving_average_comparison: bool = True
    plot_slope_of_cumulative_sum_of_moving_average_comparison: bool = True
"""

new = """    plot_online_cumulative_sum_of_moving_average_comparison: bool = True
    plot_moving_average_difference_sign_running_total: bool = True
    plot_slope_of_cumulative_sum_of_moving_average_comparison: bool = True
"""

if new not in text:
    if old not in text:
        raise RuntimeError("Could not find Settings toggle insertion point.")
    text = text.replace(old, new, 1)


# ----------------------------------------------------------------------
# 2. Add generic sign-running-total helper
# ----------------------------------------------------------------------
old = """def figure_raw_slope_sign_running_total(
    group: FigureGroup,
    settings: Settings,
):
"""

insert = """def compute_sign_running_total(
    y: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)

    sign_indicator = np.zeros(len(y), dtype=float)

    positive_mask = np.isfinite(y) & (y > 0.0)
    negative_mask = np.isfinite(y) & (y < 0.0)

    sign_indicator[positive_mask] = 1.0
    sign_indicator[negative_mask] = -1.0

    return np.cumsum(sign_indicator)


"""

new = insert + old

if insert not in text:
    if old not in text:
        raise RuntimeError("Could not find helper-function insertion point.")
    text = text.replace(old, new, 1)


# ----------------------------------------------------------------------
# 3. Add new figure function after figure_online_cumulative_ma_comparison
# ----------------------------------------------------------------------
old = """    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 3: SLOPE OF CUMULATIVE MOVING-AVERAGE COMPARISON
# ============================================================
"""

insert = """    finalize_figure(fig, axes, title)

    return fig


def figure_moving_average_difference_sign_running_total(
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
                moving_average_difference,
                _cumulative_moving_average_difference,
            ) = compute_online_moving_average_comparison_cumulative_sum(
                x,
                y,
                moving_average_window,
                baseline_moving_average_window=settings.baseline_moving_average_window,
            )

            moving_average_difference_sign_running_total = compute_sign_running_total(
                moving_average_difference
            )

            x_plot, running_total_plot = downsample_for_plotting(
                x_comparison,
                moving_average_difference_sign_running_total,
                settings,
            )

            ax.plot(
                x_plot,
                running_total_plot,
                color=get_plot_color(dataset_index, settings, "primary"),
                linewidth=2.2,
                alpha=0.95,
                label=(
                    f"{dataset.filename.stem} running total of "
                    f"sign(MA-{moving_average_window} - "
                    f"MA-{settings.baseline_moving_average_window})"
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
            ylabel=f"running total sign MA difference ({variable_label})",
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
        f"{group.title} Running Total of Moving-Average Difference Sign "
        f"[+1 if MA-{moving_average_window} > "
        f"MA-{settings.baseline_moving_average_window}, "
        f"-1 if MA-{moving_average_window} < "
        f"MA-{settings.baseline_moving_average_window}]"
    )

    finalize_figure(fig, axes, title)

    return fig


# ============================================================
# FIGURE 3: SLOPE OF CUMULATIVE MOVING-AVERAGE COMPARISON
# ============================================================
"""

if "def figure_moving_average_difference_sign_running_total(" not in text:
    if old not in text:
        raise RuntimeError("Could not find new figure-function insertion point.")
    text = text.replace(old, insert, 1)


# ----------------------------------------------------------------------
# 4. Add the new figure to PDF generation
# ----------------------------------------------------------------------
old = """    if settings.plot_slope_of_cumulative_sum_of_moving_average_comparison:
        fig = figure_slope_of_cumulative_ma_comparison(
            group=group,
            settings=settings,
            moving_average_window=moving_average_window,
        )
        add_page_to_pdf(pdf, fig, settings)
"""

new = """    if settings.plot_moving_average_difference_sign_running_total:
        fig = figure_moving_average_difference_sign_running_total(
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
"""

if new not in text:
    if old not in text:
        raise RuntimeError("Could not find PDF-generation insertion point.")
    text = text.replace(old, new, 1)


TARGET_FILE.write_text(text)
print(f"Patched {TARGET_FILE}")
