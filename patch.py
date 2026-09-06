#!/usr/bin/env python3
"""Plot FOM-rerun markers only for variables that triggered the rerun.

Usage:
    python apply_variable_specific_rerun_plot_patch_v3.py error_history.py

Expected counter comment format:
    # threshold_exceeded,iteration=15234,variables=Mass|Energy,mask=1|0|1|0

The ordinary numeric counter rows remain supported. If an old counter file has
no threshold comments, the conservation plots show no variable-specific black
markers rather than incorrectly marking every variable.
"""

import argparse
import ast
import py_compile
import re
import shutil
from pathlib import Path


NEW_LOADER = '''def load_fom_rerun_counter(counter_filename, delimiter=","):
    """Load the cumulative rerun counter and per-variable trigger records.

    Numeric rows have this format:
        iteration,cumulative_count

    Optional comment rows have this format:
        # threshold_exceeded,iteration=I,variables=...,mask=1|0|1|0

    Returns:
        counter_iterations
        counter_values
        rerun_iterations
        rerun_iterations_by_variable

    rerun_iterations_by_variable is a tuple. Entry i contains only the
    iterations at which conservative variable i exceeded the threshold.
    """
    counter_filename = Path(counter_filename)

    counter_iterations, columns = load_error_signal_file(
        counter_filename,
        delimiter=delimiter,
    )

    if columns.shape[1] != 1:
        raise ValueError(
            f"Expected one counter column in {counter_filename}; "
            f"received {columns.shape[1]}."
        )

    values_float = np.asarray(columns[:, 0], dtype=float)
    if not np.all(np.isfinite(values_float)):
        raise ValueError(
            f"Counter contains non-finite values: {counter_filename}"
        )

    rounded = np.rint(values_float)
    if not np.allclose(
        values_float,
        rounded,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"Counter contains non-integer values: {counter_filename}"
        )

    counter_values = rounded.astype(np.int64)
    changes = np.diff(counter_values, prepend=counter_values[0])

    if np.any(counter_values < 0) or np.any(changes < 0):
        raise ValueError(
            f"Counter must be nonnegative and nondecreasing: "
            f"{counter_filename}"
        )

    rerun_iterations = counter_iterations[changes > 0]

    comment_pattern = re.compile(
        r"^\\s*#\\s*threshold_exceeded,"
        r"iteration=(?P<iteration>[^,]+),"
        r"variables=(?P<variables>[^,]*),"
        r"mask=(?P<mask>[01](?:\\|[01])*)\\s*$"
    )

    trigger_records = {}
    maximum_mask_length = 0

    with counter_filename.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = comment_pattern.match(line)
            if match is None:
                continue

            try:
                iteration = float(match.group("iteration"))
            except ValueError as error:
                raise ValueError(
                    f"Invalid threshold-comment iteration on line "
                    f"{line_number} of {counter_filename}: "
                    f"{match.group('iteration')}"
                ) from error

            mask = np.asarray(
                [
                    value == "1"
                    for value in match.group("mask").split("|")
                ],
                dtype=bool,
            )

            maximum_mask_length = max(maximum_mask_length, len(mask))

            iteration_key = int(np.rint(iteration))
            if not np.isclose(
                iteration,
                iteration_key,
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"Non-integer threshold-comment iteration on line "
                    f"{line_number} of {counter_filename}: {iteration}"
                )

            if iteration_key in trigger_records:
                existing = trigger_records[iteration_key]
                if len(existing) < len(mask):
                    existing = np.pad(
                        existing,
                        (0, len(mask) - len(existing)),
                        constant_values=False,
                    )
                elif len(mask) < len(existing):
                    mask = np.pad(
                        mask,
                        (0, len(existing) - len(mask)),
                        constant_values=False,
                    )
                trigger_records[iteration_key] = existing | mask
            else:
                trigger_records[iteration_key] = mask

    # Four is the expected mass, momentum, energy, species field count.
    number_of_trigger_variables = max(4, maximum_mask_length)
    per_variable_lists = [
        []
        for _ in range(number_of_trigger_variables)
    ]

    for iteration_key, mask in sorted(trigger_records.items()):
        for variable_index, exceeded in enumerate(mask):
            if exceeded:
                per_variable_lists[variable_index].append(
                    float(iteration_key)
                )

    rerun_iterations_by_variable = tuple(
        np.asarray(values, dtype=float)
        for values in per_variable_lists
    )

    print(f"Loaded FOM-rerun counter: {counter_filename.resolve()}")
    print(f"Final count: {int(counter_values[-1])}")
    print(f"Recorded increases: {len(rerun_iterations)}")
    print(f"Threshold-comment records: {len(trigger_records)}")

    for variable_index, values in enumerate(
        rerun_iterations_by_variable
    ):
        print(
            f"Variable {variable_index} trigger count: {len(values)}"
        )

    return (
        counter_iterations,
        counter_values,
        rerun_iterations,
        rerun_iterations_by_variable,
    )
'''


def function_span(source, name):
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise RuntimeError(
            f"Expected one top-level {name}() function; found {len(nodes)}."
        )
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets[node.lineno - 1], offsets[node.end_lineno]


def replace_function(source, name, replacement):
    start, end = function_span(source, name)
    print(f"APPLY: replace {name}()")
    return source[:start] + replacement + "\n" + source[end:]


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not uniquely apply {label}; exact matches={count}."
        )
    print(f"APPLY: {label}")
    return source.replace(old, new, 1)


def replace_once_in_function(
    source,
    function_name,
    old,
    new,
    label,
):
    """Apply one exact replacement only within a named function."""
    start, end = function_span(source, function_name)
    function_source = source[start:end]
    count = function_source.count(old)

    if count != 1:
        raise RuntimeError(
            f"Could not uniquely apply {label} inside "
            f"{function_name}(); exact matches={count}."
        )

    print(f"APPLY: {label}")
    function_source = function_source.replace(old, new, 1)
    return source[:start] + function_source + source[end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_script", type=Path)
    args = parser.parse_args()

    target = args.plot_script.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Plot script not found: {target}")

    original = target.read_text(encoding="utf-8")
    ast.parse(original)

    if "rerun_iterations_by_variable" in original:
        print("The plotting script appears to be already patched.")
        return

    updated = replace_function(
        original,
        "load_fom_rerun_counter",
        NEW_LOADER,
    )

    updated = replace_once(
        updated,
        '''    fom_rerun_counter_values = None
    fom_rerun_iterations = np.array([], dtype=float)
''',
        '''    fom_rerun_counter_values = None
    fom_rerun_iterations = np.array([], dtype=float)
    fom_rerun_iterations_by_variable = tuple(
        np.array([], dtype=float)
        for _ in range(4)
    )
''',
        "initialize per-variable rerun iterations",
    )

    updated = replace_once(
        updated,
        '''                fom_rerun_counter_iterations,
                fom_rerun_counter_values,
                fom_rerun_iterations,
            ) = load_fom_rerun_counter(
''',
        '''                fom_rerun_counter_iterations,
                fom_rerun_counter_values,
                fom_rerun_iterations,
                fom_rerun_iterations_by_variable,
            ) = load_fom_rerun_counter(
''',
        "unpack per-variable trigger records",
    )

    # Both invalid-counter and missing-counter paths must reset the new data.
    reset_old = '''            fom_rerun_counter_values = None
            fom_rerun_iterations = np.array([], dtype=float)
'''
    reset_new = '''            fom_rerun_counter_values = None
            fom_rerun_iterations = np.array([], dtype=float)
            fom_rerun_iterations_by_variable = tuple(
                np.array([], dtype=float)
                for _ in range(4)
            )
'''
    updated = replace_once(
        updated,
        reset_old,
        reset_new,
        "reset per-variable records after invalid counter input",
    )

    updated = replace_once_in_function(
        updated,
        "plot_fom_rom_conservation_errors_to_pdf",
        '''    fom_rerun_iterations=None,
    short_window=1000,
''',
        '''    fom_rerun_iterations=None,
    fom_rerun_iterations_by_variable=None,
    short_window=1000,
''',
        "add per-variable trigger argument to conservation plot function",
    )

    updated = replace_once_in_function(
        updated,
        "plot_fom_rom_conservation_errors_to_pdf",
        '''    if fom_rerun_iterations is None:
        fom_rerun_iterations = np.array([], dtype=float)
    else:
        fom_rerun_iterations = np.asarray(fom_rerun_iterations, dtype=float)

    percent_file = rom_error_directory / "error_cons_perct.txt"
''',
        '''    if fom_rerun_iterations is None:
        fom_rerun_iterations = np.array([], dtype=float)
    else:
        fom_rerun_iterations = np.asarray(fom_rerun_iterations, dtype=float)

    if fom_rerun_iterations_by_variable is None:
        fom_rerun_iterations_by_variable = tuple()
    else:
        fom_rerun_iterations_by_variable = tuple(
            np.asarray(values, dtype=float)
            for values in fom_rerun_iterations_by_variable
        )

    percent_file = rom_error_directory / "error_cons_perct.txt"
''',
        "normalize per-variable trigger arrays",
    )

    updated = replace_once_in_function(
        updated,
        "plot_fom_rom_conservation_errors_to_pdf",
        '''    rerun_mask = get_fom_rerun_mask(percent_iterations, fom_rerun_iterations)

    print("============================================================")
''',
        '''    print("============================================================")
''',
        "remove global rerun mask from conservation plots",
    )

    updated = replace_once_in_function(
        updated,
        "plot_fom_rom_conservation_errors_to_pdf",
        '''    for variable_index, field_name in enumerate(field_names):
        raw = percent_values[:, variable_index]
        saved_ma = ma_values[:, variable_index]

        for plot_label, use_log_y in variants:
''',
        '''    for variable_index, field_name in enumerate(field_names):
        raw = percent_values[:, variable_index]
        saved_ma = ma_values[:, variable_index]

        if variable_index < len(fom_rerun_iterations_by_variable):
            variable_rerun_iterations = (
                fom_rerun_iterations_by_variable[variable_index]
            )
        else:
            variable_rerun_iterations = np.array([], dtype=float)

        variable_rerun_mask = get_fom_rerun_mask(
            percent_iterations,
            variable_rerun_iterations,
        )

        for plot_label, use_log_y in variants:
''',
        "create a variable-specific rerun mask",
    )

    updated = replace_once_in_function(
        updated,
        "plot_fom_rom_conservation_errors_to_pdf",
        '''            regular = finite & ~rerun_mask
            highlighted = finite & rerun_mask
''',
        '''            regular = finite & ~variable_rerun_mask
            highlighted = finite & variable_rerun_mask
''',
        "use variable-specific mask for black markers",
    )

    updated = replace_once_in_function(
        updated,
        "main",
        '''        plot_fom_rom_conservation_errors_to_pdf(
            fom_error_directory=fom_error_directory,
            rom_error_directory=rom_error_directory,
            pdf=pdf,
            fom_rerun_iterations=fom_rerun_iterations,
            short_window=args.error_short_window,
            long_window=args.error_long_window,
            delimiter=args.error_delimiter,
        )
''',
        '''        plot_fom_rom_conservation_errors_to_pdf(
            fom_error_directory=fom_error_directory,
            rom_error_directory=rom_error_directory,
            pdf=pdf,
            fom_rerun_iterations=fom_rerun_iterations,
            fom_rerun_iterations_by_variable=(
                fom_rerun_iterations_by_variable
            ),
            short_window=args.error_short_window,
            long_window=args.error_long_window,
            delimiter=args.error_delimiter,
        )
''',
        "pass per-variable triggers to conservation plots",
    )

    backup = target.with_name(
        target.name + ".before_variable_specific_rerun_plot_v3.bak"
    )
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Created backup: {backup}")
    else:
        print(f"Preserving existing backup: {backup}")

    target.write_text(updated, encoding="utf-8")
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception:
        shutil.copy2(backup, target)
        print("Syntax check failed. Restored the backup.")
        raise

    print(f"Patched file: {target}")
    print("Syntax check passed.")
    print("Black rerun markers are now variable-specific.")


if __name__ == "__main__":
    main()
