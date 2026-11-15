#!/usr/bin/env python3
from pathlib import Path
import py_compile
import sys


FUNCTION = r'''def plot_online_quarter_phase_mad_ratio(
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


'''


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor; found {count}.")
    return text.replace(old, new, 1)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} SCRIPT.py")

    path = Path(sys.argv[1]).expanduser().resolve()
    original = path.read_text(encoding="utf-8")
    updated = original

    if '"phase_quarter_mad_ratio_online"' in updated:
        raise RuntimeError("The online quarter-phase MAD patch is already applied.")

    updated = replace_once(
        updated,
        '    "phase_mad_ratio_online": True,\n',
        '    "phase_mad_ratio_online": True,\n'
        '    "phase_quarter_mad_ratio_online": True,\n',
        "quarter-phase plot toggle",
    )

    updated = replace_once(
        updated,
        '        "phase_mad_ratio_online",\n'
        '        "signed_conservation_error_cycle_sum",\n',
        '        "phase_mad_ratio_online",\n'
        '        "phase_quarter_mad_ratio_online",\n'
        '        "signed_conservation_error_cycle_sum",\n',
        "quarter-phase validation key",
    )

    anchor = "def plot_signed_conservation_error_cycle_sum("
    position = updated.find(anchor)
    if position == -1:
        raise RuntimeError("Could not find the signed cycle-sum plotting function.")
    updated = updated[:position] + FUNCTION + updated[position:]

    nested_old = (
        '                "phase_mad_ratio_online",\n'
        '                "signed_conservation_error_cycle_sum",\n'
    )
    nested_new = (
        '                "phase_mad_ratio_online",\n'
        '                "phase_quarter_mad_ratio_online",\n'
        '                "signed_conservation_error_cycle_sum",\n'
    )
    nested_count = updated.count(nested_old)
    if nested_count != 2:
        raise RuntimeError(
            "Expected the phase_mad_ratio_online nested toggle in exactly "
            f"two blocks; found {nested_count}."
        )
    updated = updated.replace(nested_old, nested_new)

    old_call = '''        if is_plot_enabled("phase_mad_ratio_online"):
            plot_online_phase_mad_ratio(
                iterations,
                raw,
                sampling_update,
                PHASE_MAD_RATIO_CYCLE_WINDOWS,
                name,
                pdf,
            )
'''
    new_call = old_call + '''
        if is_plot_enabled("phase_quarter_mad_ratio_online"):
            plot_online_quarter_phase_mad_ratio(
                iterations,
                raw,
                sampling_update,
                PHASE_MAD_RATIO_CYCLE_WINDOWS,
                name,
                pdf,
            )
'''
    updated = replace_once(updated, old_call, new_call, "quarter-phase plotting call")

    backup = path.with_suffix(path.suffix + ".before_quarter_phase_mad.bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")

    try:
        py_compile.compile(str(path), doraise=True)
    except Exception:
        path.write_text(original, encoding="utf-8")
        raise

    print(f"Patched: {path}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
