#!/usr/bin/env python3
"""Patch error_history.py so each configured iteration range gets its own PDF.

Usage:
    python patch_separate_range_pdfs.txt error_history.py

Output naming examples:
    FOM_ROM_metrics_0_to_10000.pdf
    FOM_ROM_metrics_0_to_2500.pdf
"""

import argparse
import shutil
from pathlib import Path


def replace_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected exactly one match, found {count}. "
            "No changes were written."
        )
    return source.replace(old, new, 1)


def apply_patch(source):
    # Keep the user's range setting, but select one range at a time while
    # producing each PDF.
    source = replace_once(
        source,
        ''']


def get_plot_iteration_mask(iterations):''',
        ''']


# Set internally while each range-specific PDF is generated.
ACTIVE_PLOT_ITERATION_RANGE = None


def get_plot_iteration_mask(iterations):''',
        "active range setting",
    )

    source = replace_once(
        source,
        "    if PLOT_ITERATION_RANGES is None or len(PLOT_ITERATION_RANGES) == 0:",
        "    if ACTIVE_PLOT_ITERATION_RANGE is None:",
        "active range condition",
    )

    source = replace_once(
        source,
        "    for range_index, iteration_range in enumerate(PLOT_ITERATION_RANGES):",
        "    for range_index, iteration_range in enumerate(\n"
        "        [ACTIVE_PLOT_ITERATION_RANGE]\n"
        "    ):",
        "active range loop",
    )

    # Give the current PDF a suffix matching its active inclusive range.
    source = replace_once(
        source,
        '''    field_names = [
        "Density",''',
        '''    if ACTIVE_PLOT_ITERATION_RANGE is not None:
        range_start, range_stop = ACTIVE_PLOT_ITERATION_RANGE
        output_path = Path(pdf_filename)
        pdf_filename = str(
            output_path.with_name(
                f"{output_path.stem}_{range_start:g}_to_{range_stop:g}"
                f"{output_path.suffix}"
            )
        )

    field_names = [
        "Density",''',
        "range-specific PDF filename",
    )

    # Rename the existing main routine so a small outer main can call it once
    # for every configured range without changing the plotting routines.
    source = replace_once(
        source,
        '''def main():
    """
    Main program.
    """''',
        '''def run_active_plot_range():
    """
    Generate the PDF for the currently active iteration range.
    """''',
        "existing main function rename",
    )

    source = replace_once(
        source,
        '''if __name__ == "__main__":
    main()''',
        '''def main():
    """Generate one separate PDF for every configured iteration range."""
    global ACTIVE_PLOT_ITERATION_RANGE

    if PLOT_ITERATION_RANGES is None or len(PLOT_ITERATION_RANGES) == 0:
        ranges_to_plot = [None]
    else:
        ranges_to_plot = list(PLOT_ITERATION_RANGES)

    for range_index, iteration_range in enumerate(ranges_to_plot):
        if iteration_range is not None:
            if len(iteration_range) != 2:
                raise ValueError(
                    "Each PLOT_ITERATION_RANGES entry must contain exactly "
                    f"two values. Entry {range_index}: {iteration_range}"
                )

            start, stop = iteration_range

            if not np.isfinite(start) or not np.isfinite(stop):
                raise ValueError(
                    "Plot-range limits must be finite. "
                    f"Entry {range_index}: {iteration_range}"
                )

            if start > stop:
                raise ValueError(
                    "A plot-range start cannot exceed its stop. "
                    f"Entry {range_index}: {iteration_range}"
                )

            ACTIVE_PLOT_ITERATION_RANGE = (start, stop)
            print("============================================================")
            print(
                f"Generating PDF for inclusive iteration range "
                f"{start:g} to {stop:g}"
            )
            print("============================================================")
        else:
            ACTIVE_PLOT_ITERATION_RANGE = None
            print("Generating one PDF containing all available iterations.")

        run_active_plot_range()


if __name__ == "__main__":
    main()''',
        "range-specific main loop",
    )

    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Python source file to patch")
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")
    updated = apply_patch(original)
    compile(updated, str(target), "exec")

    backup = target.with_name(target.name + ".bak")
    shutil.copy2(target, backup)
    target.write_text(updated, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("Syntax check passed.")


if __name__ == "__main__":
    main()
