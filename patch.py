#!/usr/bin/env python3
"""Remove the plot text box and put the iteration counter in the legend title.

Usage:
    python patch_iteration_legend_500cell.py path/to/animation_script.py

The target file is edited in place. A .bak copy is created before editing.
No other code is changed. The patch aborts unless both expected original
blocks occur exactly once.
"""

import argparse
import shutil
from pathlib import Path


OLD_LEGEND_AND_TEXT_BOX = """                            ax.legend(
                                loc='center left',
                                bbox_to_anchor=(1.02, 0.5),
                                borderaxespad=0,
                                fontsize=10
                            )

                            ax.grid(which='both')

                            text_box = ax.text(
                                0.02,
                                0.95,
                                (
                                    f'Case: {rom_name}\\n'
                                    f'Variable: {var_name}\\n'
                                    f'Iteration: {actual_iter}\\n'
                                    f'Percent error norm: {error_percent_norm:.6e} %'
                                ),
                                transform=ax.transAxes,
                                fontsize=11,
                                verticalalignment='top',
                                bbox=dict(
                                    boxstyle='round',
                                    facecolor='white',
                                    alpha=0.85,
                                    edgecolor='black'
                                )
                            )
"""

NEW_LEGEND_AND_TEXT_BOX = """                            legend = ax.legend(
                                loc='center left',
                                bbox_to_anchor=(1.02, 0.5),
                                borderaxespad=0,
                                fontsize=10,
                                title=f'Iteration: {actual_iter}'
                            )

                            ax.grid(which='both')
"""

OLD_UPDATE = """                            text_box.set_text(
                                (
                                    f'Case: {rom_name}\\n'
                                    f'Variable: {var_name}\\n'
                                    f'Iteration: {actual_iter}\\n'
                                    f'Percent error norm: {error_percent_norm:.6e} %'
                                )
                            )
"""

NEW_UPDATE = """                            legend.set_title(
                                f'Iteration: {actual_iter}'
                            )
"""


def replace_exactly_once(source, old, new, description):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {description} block, but found {count}. "
            "No changes were written."
        )
    return source.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove the plot information box and retain the current iteration "
            "as the legend title."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Python animation script to patch in place",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist: {target}")

    original = target.read_text(encoding="utf-8")

    patched = replace_exactly_once(
        original,
        OLD_LEGEND_AND_TEXT_BOX,
        NEW_LEGEND_AND_TEXT_BOX,
        "legend/text-box",
    )
    patched = replace_exactly_once(
        patched,
        OLD_UPDATE,
        NEW_UPDATE,
        "text-box update",
    )

    backup = target.with_name(target.name + ".bak")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    print("Removed the plot text box and added the iteration to the legend title.")


if __name__ == "__main__":
    main()
