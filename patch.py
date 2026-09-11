#!/usr/bin/env python3
"""Add rolling spread and CV figures based on each winsorized mean.

Apply to the exact code version supplied with this request.

Usage:
    python patch_winsorized_std_cv.py error_history.py

The target is modified in place after syntax validation. A one-time backup is
created as ``error_history.py.before_winsorized_std_cv.bak``.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count < 1:
        raise RuntimeError(
            f"Could not apply {label}: expected exactly one match, found {count}. "
            "Apply this patch to the exact code version supplied with the request."
        )
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Add rolling winsorized-mean spread and CV plots."
    )
    parser.add_argument("target", type=Path, help="Plotting Python file to patch")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target Python file does not exist: {target}")

    updated = target.read_text(encoding="utf-8")

    updated = replace_once(
        updated,
        '    "winsorized_residual_metrics": True,\n'
        '    "adaptive_valley_band": False,',
        '    "winsorized_residual_metrics": True,\n'
        '    "winsorized_standard_deviation": True,\n'
        '    "winsorized_coefficient_of_variation": True,\n'
        '    "adaptive_valley_band": False,',
        "plot switches",
    )

    updated = replace_once(
        updated,
        'WINSORIZED_RESIDUAL_AVERAGE_WINDOW = 1000\n'
        'WINSORIZED_RATIO_MIN_DENOMINATOR = 1.0e-30\n',
        'WINSORIZED_RESIDUAL_AVERAGE_WINDOW = 1000\n'
        'WINSORIZED_RATIO_MIN_DENOMINATOR = 1.0e-30\n'
        'WINSORIZED_STD_CV_WINDOW = 1000\n'
        'WINSORIZED_CV_MIN_BASELINE_MAGNITUDE = 1.0e-30\n'
        'WINSORIZED_STD_LOGY = True\n'
        'WINSORIZED_CV_LOGY = True\n',
        "winsorized spread settings",
    )

    updated = replace_once(
        updated,
        '        "winsorized_residual_metrics", "adaptive_valley_band", "rolling_rms",',
        '        "winsorized_residual_metrics", "winsorized_standard_deviation",\n'
        '        "winsorized_coefficient_of_variation", "adaptive_valley_band",\n'
        '        "rolling_rms",',
        "allowed plot keys",
    )

    updated = replace_once(
        updated,
        '    require_nonnegative(\n'
        '        WINSORIZED_RATIO_MIN_DENOMINATOR,\n'
        '        "WINSORIZED_RATIO_MIN_DENOMINATOR",\n'
        '    )\n'
        '    if not WINSORIZED_TAIL_FRACTIONS:',
        '    require_nonnegative(\n'
        '        WINSORIZED_RATIO_MIN_DENOMINATOR,\n'
        '        "WINSORIZED_RATIO_MIN_DENOMINATOR",\n'
        '    )\n'
        '    require_positive(WINSORIZED_STD_CV_WINDOW, "WINSORIZED_STD_CV_WINDOW")\n'
        '    require_nonnegative(\n'
        '        WINSORIZED_CV_MIN_BASELINE_MAGNITUDE,\n'
        '        "WINSORIZED_CV_MIN_BASELINE_MAGNITUDE",\n'
        '    )\n'
        '    if not isinstance(WINSORIZED_STD_LOGY, bool):\n'
        '        raise TypeError("WINSORIZED_STD_LOGY must be True or False.")\n'
        '    if not isinstance(WINSORIZED_CV_LOGY, bool):\n'
        '        raise TypeError("WINSORIZED_CV_LOGY must be True or False.")\n'
        '    if not WINSORIZED_TAIL_FRACTIONS:',
        "winsorized spread validation",
    )

    analysis_code = '''\n\ndef rolling_winsorized_std_and_cv(
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
'''

    updated = replace_once(
        updated,
        "\n\ndef exponential_moving_average(values, span):",
        analysis_code + "\n\ndef exponential_moving_average(values, span):",
        "winsorized std/CV analysis function",
    )

    plotting_code = '''\n\ndef plot_winsorized_std_cv_pages(
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
            f"{title}: {name}\\nRolling spread about winsorized means "
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
            f"{title}: {name}\\nRolling coefficient of variation based on "
            f"winsorized means (window={WINSORIZED_STD_CV_WINDOW})",
            "Rolling CV = spread / |winsorized mean|",
            pdf,
            logy=WINSORIZED_CV_LOGY,
        )
'''

    updated = replace_once(
        updated,
        "\n\ndef plot_baseline_pages(iterations, raw, title, name, pdf, baseline_window, rms_window):",
        plotting_code + "\n\ndef plot_baseline_pages(iterations, raw, title, name, pdf, baseline_window, rms_window):",
        "winsorized std/CV plotting function",
    )

    old_call = '''        if is_plot_enabled("winsorized_residual_metrics"):
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
    if is_plot_enabled("adaptive_valley_band"):'''

    new_call = '''        if is_plot_enabled("winsorized_residual_metrics"):
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
    if is_plot_enabled("adaptive_valley_band"):'''

    updated = replace_once(
        updated,
        old_call,
        new_call,
        "winsorized std/CV plot invocation",
    )

    updated = replace_once(
        updated,
        '            "winsorized_residual_metrics", "adaptive_valley_band", "rolling_rms",',
        '            "winsorized_residual_metrics", "winsorized_standard_deviation",\n'
        '            "winsorized_coefficient_of_variation", "adaptive_valley_band",\n'
        '            "rolling_rms",',
        "plot orchestration gate",
    )

    ast.parse(updated, filename=str(target))

    backup = target.with_suffix(target.suffix + ".before_winsorized_std_cv.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(updated, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
