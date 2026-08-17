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
    # Saved moving-average figures
    # ----------------------------
    plot_saved_moving_average_overlay_data: bool = True
    plot_moving_average_counter_data: bool = True

    short_ma_suffix: str = "_short_ma"
    long_ma_suffix: str = "_long_ma"
    ma_counter_suffix: str = "_ma_counter"
'''

    new = '''    # ----------------------------
    # Saved moving-average figures
    # ----------------------------
    plot_saved_moving_average_overlay_data: bool = True
    plot_moving_average_counter_data: bool = True
    plot_moving_average_counter_sum_data: bool = True

    short_ma_suffix: str = "_short_ma"
    long_ma_suffix: str = "_long_ma"
    ma_counter_suffix: str = "_ma_counter"
    ma_counter_sum_suffix: str = "_ma_counter_sum"
'''

    return replace_once(
        text,
        old,
        new,
        "Settings moving-average counter sum options",
    )


def patch_filename_helper(text: str) -> str:
    old = '''def get_ma_counter_filename(filename: Path, settings: Settings) -> Path:
    return filename.with_name(
        f"{filename.stem}{settings.ma_counter_suffix}{filename.suffix}"
    )


# ============================================================
# COMPUTATION HELPERS
# ============================================================
'''

    new = '''def get_ma_counter_filename(filename: Path, settings: Settings) -> Path:
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
'''

    return replace_once(
        text,
        old,
        new,
        "MA counter sum filename helper",
    )


def patch_counter_sum_figure(text: str) -> str:
    old = '''# ============================================================
# FIGURE 2: SLOPE-COUNTER RAW DATA
# (same variable layout as the group, loaded from
#  "<original_filename>_slope_counter.txt")
# ============================================================
'''

    new = '''# ============================================================
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
'''

    return replace_once(
        text,
        old,
        new,
        "MA counter sum figure function",
    )


def patch_add_enabled_figures_for_group(text: str) -> str:
    old = '''    if settings.plot_moving_average_counter_data:
        fig = figure_moving_average_counter_data(
            group=group,
            settings=settings,
        )
        add_page_to_pdf(pdf, fig, settings)

    fig = figure_main_data(
'''

    new = '''    if settings.plot_moving_average_counter_data:
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
'''

    return replace_once(
        text,
        old,
        new,
        "add MA counter sum figure to add_enabled_figures_for_group",
    )


def apply_patch(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find file: {script_path}")

    original_text = script_path.read_text()

    if "plot_moving_average_counter_sum_data" in original_text:
        raise RuntimeError(
            "Patch appears to already be applied. "
            "Found plot_moving_average_counter_sum_data in the file."
        )

    patched_text = original_text
    patched_text = patch_settings(patched_text)
    patched_text = patch_filename_helper(patched_text)
    patched_text = patch_counter_sum_figure(patched_text)
    patched_text = patch_add_enabled_figures_for_group(patched_text)

    backup_path = script_path.with_suffix(script_path.suffix + ".bak")
    backup_path.write_text(original_text)
    script_path.write_text(patched_text)

    print(f"Patched file: {script_path}")
    print(f"Backup saved to: {backup_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch plotting script to add moving-average counter sum figure."
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