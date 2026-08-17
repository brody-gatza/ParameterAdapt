#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)

    if count == 0:
        raise RuntimeError(f"Could not find patch location for: {label}")

    if count > 1:
        raise RuntimeError(f"Found multiple patch locations for: {label}")

    return text.replace(old, new, 1)


def patch_settings(text: str) -> str:
    old = '''    # ----------------------------
    # Slope-counter / slope-ratio overlay figure
    # ----------------------------
    plot_slope_counter_ratio_overlay_data: bool = True
'''

    new = '''    # ----------------------------
    # Slope-counter / slope-ratio overlay figure
    # ----------------------------
    plot_slope_counter_ratio_overlay_data: bool = True

    # ----------------------------
    # Saved moving-average figures
    # ----------------------------
    plot_saved_moving_average_overlay_data: bool = True
    plot_moving_average_counter_data: bool = True

    short_ma_suffix: str = "_short_ma"
    long_ma_suffix: str = "_long_ma"
    ma_counter_suffix: str = "_ma_counter"
'''

    return replace_once(text, old, new, "Settings moving-average options")


def patch_filename_helpers(text: str) -> str:
    old = '''def get_slope_ratio_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.slope_ratio_suffix}{filename.suffix}"
    )


# ============================================================
# COMPUTATION HELPERS
# ============================================================
'''

    new = '''def get_slope_ratio_filename(filename: Path, settings: Settings) -> Path:
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


# ============================================================
# COMPUTATION HELPERS
# ============================================================
'''

    return replace_once(text, old, new, "moving-average filename helpers")


def patch_new_figure_functions(text: str) -> str:
    old = '''# ============================================================
# FIGURE 2: SLOPE-COUNTER RAW DATA
# (same variable layout as the group, loaded from
#  "<original_filename>_slope_counter.txt")
# ============================================================
'''

    new = '''# ============================================================
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
# FIGURE 2: SLOPE-COUNTER RAW DATA
# (same variable layout as the group, loaded from
#  "<original_filename>_slope_counter.txt")
# ============================================================
'''

    return replace_once(text, old, new, "saved MA figure functions")


def patch_add_enabled_figures_for_group(text: str) -> str:
    old = '''def add_enabled_figures_for_group(
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
'''

    new = '''def add_enabled_figures_for_group(
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
'''

    return replace_once(text, old, new, "add_enabled_figures_for_group")


def apply_patch(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find file: {script_path}")

    text = script_path.read_text()

    if "plot_saved_moving_average_overlay_data" in text:
        raise RuntimeError(
            "Patch appears to have already been applied. "
            "Found plot_saved_moving_average_overlay_data in the file."
        )

    patched = text
    patched = patch_settings(patched)
    patched = patch_filename_helpers(patched)
    patched = patch_new_figure_functions(patched)
    patched = patch_add_enabled_figures_for_group(patched)

    backup_path = script_path.with_suffix(script_path.suffix + ".bak")
    backup_path.write_text(text)
    script_path.write_text(patched)

    print(f"Patched file: {script_path}")
    print(f"Backup saved: {backup_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch plotting script to add saved moving-average figures."
    )

    parser.add_argument(
        "script",
        help="Path to the plotting script to patch.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_patch(Path(args.script))


if __name__ == "__main__":
    main()
