#!/usr/bin/env python3
"""Compare conservation-error histories across any number of AROM cases.

Each case argument may be either:
  1. a case directory named like s0.5_u2, containing
     AROM_results/error/error_cons.txt; or
  2. the error_cons.txt file itself, with an ancestor directory named s#_u#.

For each conservative variable, the output multipage PDF contains:
  1. all raw conservation-error points for every case;
  2. only the first error point after each sampling update; and
  3. the causal normalized MAD over approximately 100 simulation iterations,
     computed from those selected points.

Sampling-update points are 11 + k*u, where u is parsed from the case name.
The MAD is computed from the selected post-update dataset, not the full raw
history. For each case, the selected-point count is floor(100 / u) + 1, so the
endpoint-to-endpoint span does not exceed 100 simulation iterations.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


CONSERVATIVE_FIELD_NAMES = ["Mass", "Momentum", "Energy", "Species"]
CASE_NAME_PATTERN = re.compile(
    r"^s(?P<s>[+-]?(?:\d+(?:\.\d*)?|\.\d+))_u"
    r"(?P<u>[+-]?(?:\d+(?:\.\d*)?|\.\d+))$",
    re.IGNORECASE,
)
DEFAULT_ERROR_FILENAME = "error_cons.txt"
FIRST_POST_UPDATE_ITERATION = 11.0
MAD_WINDOW_ITERATIONS = 100.0
MAD_MIN_BASELINE_MAGNITUDE = 1.0e-30


@dataclass(frozen=True)
class CaseData:
    label: str
    source: Path
    update_frequency: float
    iterations: np.ndarray
    errors: np.ndarray


def positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw conservation errors and post-update normalized MAD "
            "for cases named s#_u#, saving all figures in one PDF."
        )
    )
    parser.add_argument(
        "cases",
        nargs="+",
        type=Path,
        help=(
            "Case directories or error_cons.txt files. A case name or one of "
            "its parent directories must have the form s#_u#, such as s0.5_u2."
        ),
    )
    parser.add_argument(
        "--iteration-range",
        "--range",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        required=True,
        help="Inclusive iteration range to plot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("case_conservation_error_comparison.pdf"),
        help="Output multipage PDF (default: %(default)s).",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Input delimiter for error_cons.txt (default: comma).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear axes for raw-error figures. Positive errors use log by default.",
    )
    parser.add_argument(
        "--dpi",
        type=positive_float,
        default=150.0,
        help="Rasterization DPI used when saving PDF pages (default: 150).",
    )
    return parser.parse_args()


def validate_iteration_range(
    values: list[float] | tuple[float, float],
) -> tuple[float, float]:
    start, stop = map(float, values)
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("Iteration-range bounds must be finite.")
    if start > stop:
        raise ValueError(
            f"Iteration-range start ({start:g}) exceeds stop ({stop:g})."
        )
    return start, stop


def find_case_name(path: Path) -> tuple[str, float]:
    candidates = [path.stem if path.is_file() else path.name]
    candidates.extend(parent.name for parent in path.parents)

    for candidate in candidates:
        match = CASE_NAME_PATTERN.fullmatch(candidate)
        if match:
            update_frequency = float(match.group("u"))
            if not np.isfinite(update_frequency) or update_frequency <= 0.0:
                raise ValueError(
                    f"Sampling update frequency in {candidate!r} must be positive."
                )
            return candidate, update_frequency

    raise ValueError(
        f"Could not find a case name of the form s#_u# in or above: {path}"
    )


def resolve_error_file(case_path: Path) -> Path:
    path = case_path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Case path does not exist: {path}")

    candidates = (
        path / "AROM_results" / "error" / DEFAULT_ERROR_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find error_cons.txt. Checked:\n"
        + "\n".join(f"  {candidate}" for candidate in candidates)
    )


def load_case(
    case_path: Path,
    active_range: tuple[float, float],
    delimiter: str,
) -> CaseData:
    error_file = resolve_error_file(case_path)
    label, update_frequency = find_case_name(case_path.expanduser().resolve())

    data = np.loadtxt(error_file, delimiter=delimiter, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected iteration plus at least one error column in {error_file}; "
            f"received shape {data.shape}."
        )

    iterations = np.asarray(data[:, 0], dtype=float)
    errors = np.abs(np.asarray(data[:, 1:], dtype=float))
    if not np.all(np.isfinite(iterations)):
        raise ValueError(f"Non-finite iteration values found in {error_file}.")

    order = np.argsort(iterations, kind="stable")
    iterations = iterations[order]
    errors = errors[order]
    if np.any(np.diff(iterations) == 0.0):
        raise ValueError(f"Duplicate iterations found in {error_file}.")

    start, stop = active_range
    mask = (iterations >= start) & (iterations <= stop)
    if not np.any(mask):
        raise ValueError(
            f"No data from {label} fall in iteration range {start:g} to {stop:g}."
        )

    return CaseData(
        label=label,
        source=error_file,
        update_frequency=update_frequency,
        iterations=iterations[mask],
        errors=errors[mask],
    )


def conservative_names(count: int) -> list[str]:
    if count == len(CONSERVATIVE_FIELD_NAMES):
        return list(CONSERVATIVE_FIELD_NAMES)
    return [
        CONSERVATIVE_FIELD_NAMES[index]
        if index < len(CONSERVATIVE_FIELD_NAMES)
        else f"Conservative field {index + 1}"
        for index in range(count)
    ]


def apply_standard_axes(
    axis: plt.Axes,
    use_log_y: bool,
    ylabel: str = "Absolute conservation error",
) -> None:
    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.minorticks_on()
    axis.grid(which="major", axis="both", linestyle="-", alpha=0.35)
    axis.grid(which="minor", axis="x", linestyle=":", alpha=0.20)
    if use_log_y:
        axis.set_yscale("log")


def can_use_log_scale(series: list[np.ndarray]) -> bool:
    finite_values = [
        values[np.isfinite(values)]
        for values in series
        if np.any(np.isfinite(values))
    ]
    return bool(finite_values) and all(
        values.size > 0 and np.all(values > 0.0) for values in finite_values
    )


def first_post_update_mask(
    iterations: np.ndarray,
    update_frequency: float,
) -> np.ndarray:
    """Select stored iterations matching 11 + k*u."""
    iterations = np.asarray(iterations, dtype=float)
    offset = iterations - FIRST_POST_UPDATE_ITERATION
    cycle = np.rint(offset / update_frequency)
    expected = FIRST_POST_UPDATE_ITERATION + cycle * update_frequency
    tolerance = max(1.0e-9, 1.0e-9 * update_frequency)
    return (cycle >= 0.0) & np.isclose(
        iterations,
        expected,
        rtol=0.0,
        atol=tolerance,
    )


def rolling_normalized_mad(
    values: np.ndarray,
    window: int,
    minimum_baseline_magnitude: float = MAD_MIN_BASELINE_MAGNITUDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the old code's causal median, MAD, and normalized MAD.

    For selected-sample index i, use selected values
    max(0, i-window+1) through i. Non-finite values are removed.

        median = median(window_values)
        MAD = median(abs(window_values - median))
        normalized_MAD = MAD / abs(median)

    The first window-1 results use all selected samples available so far,
    matching the behavior of rolling_median_spread_statistics in the old code.
    """
    values = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("MAD window must be positive.")
    if minimum_baseline_magnitude < 0.0:
        raise ValueError("minimum_baseline_magnitude must be nonnegative.")

    medians = np.full(values.shape, np.nan, dtype=float)
    mads = np.full(values.shape, np.nan, dtype=float)
    normalized_mads = np.full(values.shape, np.nan, dtype=float)

    for index in range(len(values)):
        current = values[max(0, index - window + 1):index + 1]
        current = current[np.isfinite(current)]
        if current.size == 0:
            continue

        center = np.median(current)
        mad = np.median(np.abs(current - center))
        medians[index] = center
        mads[index] = mad
        if abs(center) > minimum_baseline_magnitude:
            normalized_mads[index] = mad / abs(center)

    return medians, mads, normalized_mads


def save_figure(pdf: PdfPages, figure: plt.Figure, dpi: float) -> None:
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight", dpi=dpi)
    plt.close(figure)


def plot_raw_comparison(
    cases: list[CaseData],
    variable_index: int,
    variable_name: str,
    active_range: tuple[float, float],
    pdf: PdfPages,
    linear_y: bool,
    dpi: float,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = []
    for case in cases:
        values = case.errors[:, variable_index]
        axis.plot(
            case.iterations,
            values,
            linewidth=1.35,
            alpha=0.9,
            label=case.label,
        )
        plotted.append(values)

    use_log_y = not linear_y and can_use_log_scale(plotted)
    apply_standard_axes(axis, use_log_y)
    start, stop = active_range
    axis.set_title(
        f"Raw Conservation Error: {variable_name}\n"
        f"Iterations {start:g} to {stop:g}"
    )
    axis.legend(loc="best", title="Case")
    save_figure(pdf, figure, dpi)


def plot_first_post_update_comparison(
    cases: list[CaseData],
    variable_index: int,
    variable_name: str,
    active_range: tuple[float, float],
    pdf: PdfPages,
    linear_y: bool,
    dpi: float,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    plotted = []

    for case in cases:
        mask = first_post_update_mask(case.iterations, case.update_frequency)
        selected_iterations = case.iterations[mask]
        selected_values = case.errors[mask, variable_index]
        if selected_iterations.size == 0:
            print(
                f"Warning: {case.label} has no stored first-post-update points "
                f"in the selected range. Expected 11 + k*{case.update_frequency:g}."
            )
            continue

        axis.plot(
            selected_iterations,
            selected_values,
            marker="o",
            markersize=3.5,
            linewidth=1.25,
            alpha=0.9,
            label=f"{case.label} (u={case.update_frequency:g})",
        )
        plotted.append(selected_values)

    if not plotted:
        plt.close(figure)
        raise RuntimeError(
            f"No first-post-update points were found for {variable_name}."
        )

    use_log_y = not linear_y and can_use_log_scale(plotted)
    apply_standard_axes(axis, use_log_y)
    start, stop = active_range
    axis.set_title(
        f"First Conservation Error Point After Each Sampling Update: {variable_name}\n"
        f"Iterations {start:g} to {stop:g}; selected points = 11 + k*u"
    )
    axis.legend(loc="best", title="Case")
    save_figure(pdf, figure, dpi)


def plot_post_update_mad_comparison(
    cases: list[CaseData],
    variable_index: int,
    variable_name: str,
    active_range: tuple[float, float],
    pdf: PdfPages,
    dpi: float,
) -> None:
    """Plot normalized MAD computed only from first-post-update points."""
    figure, axis = plt.subplots(figsize=(12, 7))
    plotted_count = 0

    for case in cases:
        mask = first_post_update_mask(case.iterations, case.update_frequency)
        selected_iterations = case.iterations[mask]
        selected_values = case.errors[mask, variable_index]
        if selected_iterations.size == 0:
            continue

        # Convert the common 100-iteration span into a number of selected
        # post-update data points for this case. The +1 includes both ends:
        # u=2 -> 51 points spanning 100 iterations
        # u=5 -> 21 points spanning 100 iterations
        # u=20 -> 6 points spanning 100 iterations
        mad_window_points = max(
            1,
            int(np.floor(
                MAD_WINDOW_ITERATIONS / case.update_frequency
            )) + 1,
        )

        _, _, normalized_mad = rolling_normalized_mad(
            selected_values,
            window=mad_window_points,
        )
        finite = np.isfinite(normalized_mad)
        if not np.any(finite):
            print(
                f"Warning: {case.label} produced no finite normalized MAD "
                f"values for {variable_name}."
            )
            continue

        axis.plot(
            selected_iterations[finite],
            normalized_mad[finite],
            linewidth=1.8,
            alpha=0.95,
            label=(
                f"{case.label} (u={case.update_frequency:g}, "
                f"window={mad_window_points} points)"
            ),
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(figure)
        raise RuntimeError(
            f"No finite post-update normalized MAD data were found for {variable_name}."
        )

    apply_standard_axes(
        axis,
        use_log_y=False,
        ylabel="Normalized MAD = MAD / |median|",
    )
    axis.set_ylim(bottom=0.0)
    start, stop = active_range
    axis.set_title(
        f"Post-Update Conservation Error Normalized MAD: {variable_name}\n"
        f"Causal span = {MAD_WINDOW_ITERATIONS:g} simulation iterations; "
        f"iterations {start:g} to {stop:g}"
    )
    axis.legend(loc="best", title="Case")
    save_figure(pdf, figure, dpi)


def main() -> None:
    args = parse_arguments()
    active_range = validate_iteration_range(args.iteration_range)
    cases = [load_case(path, active_range, args.delimiter) for path in args.cases]

    column_counts = {case.errors.shape[1] for case in cases}
    if len(column_counts) != 1:
        details = ", ".join(
            f"{case.label}: {case.errors.shape[1]}" for case in cases
        )
        raise ValueError(
            "All cases must contain the same number of conservation-error "
            f"columns. Received {details}."
        )

    variable_count = column_counts.pop()
    names = conservative_names(variable_count)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Output PDF: {output}")
    print(f"Iteration range: {active_range[0]:g} to {active_range[1]:g}")
    print(
        f"Post-update MAD span: {MAD_WINDOW_ITERATIONS:g} simulation iterations"
    )
    for case in cases:
        selected_count = int(
            np.count_nonzero(
                first_post_update_mask(case.iterations, case.update_frequency)
            )
        )
        print(
            f"{case.label}: u={case.update_frequency:g}, "
            f"raw rows={len(case.iterations)}, selected rows={selected_count}, "
            f"file={case.source}"
        )
    print("=" * 72)

    with PdfPages(output) as pdf:
        for variable_index, variable_name in enumerate(names):
            plot_raw_comparison(
                cases,
                variable_index,
                variable_name,
                active_range,
                pdf,
                args.linear_y,
                args.dpi,
            )
            plot_first_post_update_comparison(
                cases,
                variable_index,
                variable_name,
                active_range,
                pdf,
                args.linear_y,
                args.dpi,
            )
            plot_post_update_mad_comparison(
                cases,
                variable_index,
                variable_name,
                active_range,
                pdf,
                args.dpi,
            )

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
