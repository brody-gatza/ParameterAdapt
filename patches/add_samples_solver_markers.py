from pathlib import Path

target_file = Path("animate.py")
backup_file = Path("animate.py.bak")

if not target_file.exists():
    raise FileNotFoundError(f"Could not find {target_file.resolve()}")

text = target_file.read_text()

if not backup_file.exists():
    backup_file.write_text(text)
    print(f"Backup written to {backup_file}")
else:
    print(f"Backup already exists at {backup_file}")

def replace_once(text, old, new, description):
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"Could not find insertion point for: {description}")
    if count > 1:
        raise RuntimeError(f"Found multiple insertion points for: {description}")
    return text.replace(old, new, 1)

# ----------------------------------------------------------------------
# 1. Add helper function to load samples_solver files
# ----------------------------------------------------------------------
old = '''def read_prim_data(args):
    iter_num, result_dir = args
    filename = os.path.join(result_dir, f"{iter_num}iteration_prim.npy")
    return np.load(filename)
'''

new = '''def read_prim_data(args):
    iter_num, result_dir = args
    filename = os.path.join(result_dir, f"{iter_num}iteration_prim.npy")
    return np.load(filename)


def load_samples_solver_cells(samples_solver_dir, iter_num, cell_num):
    """
    Loads sampled solver cell numbers for one iteration.

    Expected file format:
        {iter_num}iteration_samples_solver.npy

    The file is expected to contain a 1D array of integer cell numbers.
    Only values satisfying:
        0 <= cell_number < cell_num
    are returned.

    Returned values are valid indices into:
        x[cell_number]
        state_vector[cell_number]
    """

    filename = os.path.join(
        samples_solver_dir,
        f"{iter_num}iteration_samples_solver.npy"
    )

    if not os.path.exists(filename):
        return np.array([], dtype=int)

    sample_cells = np.load(filename)
    sample_cells = np.asarray(sample_cells).ravel().astype(int)

    sample_cells = sample_cells[
        (sample_cells >= 0) &
        (sample_cells < cell_num)
    ]

    return sample_cells
'''

if "def load_samples_solver_cells(" not in text:
    text = replace_once(text, old, new, "load_samples_solver_cells")
else:
    print("load_samples_solver_cells already present, skipping helper insertion")

# ----------------------------------------------------------------------
# 2. Update build_rom_case_paths signature
# ----------------------------------------------------------------------
old = '''def build_rom_case_paths(
    base_directory,
    rom_case_names,
    rom_result_subdir,
    animation_output_subdir
):
'''

new = '''def build_rom_case_paths(
    base_directory,
    rom_case_names,
    rom_result_subdir,
    samples_solver_subdir,
    animation_output_subdir
):
'''

if "samples_solver_subdir," not in text[text.find("def build_rom_case_paths("):text.find("def build_rom_case_paths(") + 300]:
    text = replace_once(text, old, new, "build_rom_case_paths signature")
else:
    print("build_rom_case_paths signature already updated, skipping")

# ----------------------------------------------------------------------
# 3. Update build_rom_case_paths docstring return list
# ----------------------------------------------------------------------
old = '''    Returns:
        rom_names
        rom_case_dirs
        rom_result_dirs
        rom_animation_base_dirs
'''

new = '''    Returns:
        rom_names
        rom_case_dirs
        rom_result_dirs
        rom_samples_solver_dirs
        rom_animation_base_dirs
'''

if "rom_samples_solver_dirs" not in text[text.find("def build_rom_case_paths("):text.find("def build_rom_case_paths(") + 1200]:
    text = replace_once(text, old, new, "build_rom_case_paths docstring")
else:
    print("rom_samples_solver_dirs already present in build_rom_case_paths, skipping docstring update")

# ----------------------------------------------------------------------
# 4. Add rom_samples_solver_dirs list
# ----------------------------------------------------------------------
old = '''    rom_names = []
    rom_case_dirs = []
    rom_result_dirs = []
    rom_animation_base_dirs = []
'''

new = '''    rom_names = []
    rom_case_dirs = []
    rom_result_dirs = []
    rom_samples_solver_dirs = []
    rom_animation_base_dirs = []
'''

if "rom_samples_solver_dirs = []" not in text:
    text = replace_once(text, old, new, "rom_samples_solver_dirs list")
else:
    print("rom_samples_solver_dirs list already present, skipping")

# ----------------------------------------------------------------------
# 5. Add samples_solver path in build_rom_case_paths
# ----------------------------------------------------------------------
old = '''        rom_result_dir = os.path.join(
            rom_case_dir,
            rom_result_subdir
        )

        rom_animation_base_dir = os.path.join(
            rom_case_dir,
            animation_output_subdir
        )
'''

new = '''        rom_result_dir = os.path.join(
            rom_case_dir,
            rom_result_subdir
        )

        rom_samples_solver_dir = os.path.join(
            rom_case_dir,
            samples_solver_subdir
        )

        rom_animation_base_dir = os.path.join(
            rom_case_dir,
            animation_output_subdir
        )
'''

if "rom_samples_solver_dir = os.path.join" not in text:
    text = replace_once(text, old, new, "rom_samples_solver_dir path")
else:
    print("rom_samples_solver_dir path already present, skipping")

# ----------------------------------------------------------------------
# 6. Append samples_solver path to list
# ----------------------------------------------------------------------
old = '''        rom_names.append(rom_name)
        rom_case_dirs.append(rom_case_dir)
        rom_result_dirs.append(rom_result_dir)
        rom_animation_base_dirs.append(rom_animation_base_dir)
'''

new = '''        rom_names.append(rom_name)
        rom_case_dirs.append(rom_case_dir)
        rom_result_dirs.append(rom_result_dir)
        rom_samples_solver_dirs.append(rom_samples_solver_dir)
        rom_animation_base_dirs.append(rom_animation_base_dir)
'''

if "rom_samples_solver_dirs.append(rom_samples_solver_dir)" not in text:
    text = replace_once(text, old, new, "rom_samples_solver_dirs append")
else:
    print("rom_samples_solver_dirs append already present, skipping")

# ----------------------------------------------------------------------
# 7. Update build_rom_case_paths return
# ----------------------------------------------------------------------
old = '''    return rom_names, rom_case_dirs, rom_result_dirs, rom_animation_base_dirs
'''

new = '''    return (
        rom_names,
        rom_case_dirs,
        rom_result_dirs,
        rom_samples_solver_dirs,
        rom_animation_base_dirs
    )
'''

if "rom_samples_solver_dirs," not in text[text.find("return ("):text.find("return (") + 300]:
    text = replace_once(text, old, new, "build_rom_case_paths return")
else:
    print("build_rom_case_paths return already updated, skipping")

# ----------------------------------------------------------------------
# 8. Add samples_solver_subdir setting after rom_result_subdir
# ----------------------------------------------------------------------
old = '''    rom_result_subdir = os.path.join('AROM_results', 'cons_prim')
'''

new = '''    rom_result_subdir = os.path.join('AROM_results', 'cons_prim')

    # This is where the sampled solver cell files are located inside each case.
    #
    # Expected file format:
    #     /.../case_name/AROM_results/samples_solver/{iter}iteration_samples_solver.npy
    #
    # Each file should contain a 1D integer array of cell numbers.
    samples_solver_subdir = os.path.join('AROM_results', 'samples_solver')
'''

if "samples_solver_subdir = os.path.join('AROM_results', 'samples_solver')" not in text:
    text = replace_once(text, old, new, "samples_solver_subdir setting")
else:
    print("samples_solver_subdir already present, skipping")

# ----------------------------------------------------------------------
# 9. Update build_rom_case_paths call and unpacking
# ----------------------------------------------------------------------
old = '''    rom_names, rom_case_dirs, rom_result_dirs, rom_animation_base_dirs = build_rom_case_paths(
        base_directory=directory,
        rom_case_names=rom_case_names,
        rom_result_subdir=rom_result_subdir,
        animation_output_subdir=animation_output_subdir
    )
'''

new = '''    (
        rom_names,
        rom_case_dirs,
        rom_result_dirs,
        rom_samples_solver_dirs,
        rom_animation_base_dirs
    ) = build_rom_case_paths(
        base_directory=directory,
        rom_case_names=rom_case_names,
        rom_result_subdir=rom_result_subdir,
        samples_solver_subdir=samples_solver_subdir,
        animation_output_subdir=animation_output_subdir
    )
'''

if "rom_samples_solver_dirs," not in text[text.find(") = build_rom_case_paths(") - 200:text.find(") = build_rom_case_paths(") + 500]:
    text = replace_once(text, old, new, "build_rom_case_paths call")
else:
    print("build_rom_case_paths call already updated, skipping")

# ----------------------------------------------------------------------
# 10. Update ROM case loop zip
# ----------------------------------------------------------------------
old = '''        for rom_name, rom_result_dir, rom_animation_base_dir in zip(
            rom_names,
            rom_result_dirs,
            rom_animation_base_dirs
        ):
'''

new = '''        for (
            rom_name,
            rom_result_dir,
            rom_samples_solver_dir,
            rom_animation_base_dir
        ) in zip(
            rom_names,
            rom_result_dirs,
            rom_samples_solver_dirs,
            rom_animation_base_dirs
        ):
'''

if "rom_samples_solver_dir," not in text[text.find("for ("):text.find("for (") + 500]:
    text = replace_once(text, old, new, "ROM case loop zip")
else:
    print("ROM case loop already updated, skipping")

# ----------------------------------------------------------------------
# 11. Add warning if samples_solver directory is missing
# ----------------------------------------------------------------------
old = '''            if not os.path.isdir(rom_result_dir):
                print(f"Warning: ROM directory does not exist. Skipping {rom_name}.")
                print(f"  Missing directory: {rom_result_dir}")
                continue
'''

new = '''            if not os.path.isdir(rom_result_dir):
                print(f"Warning: ROM directory does not exist. Skipping {rom_name}.")
                print(f"  Missing directory: {rom_result_dir}")
                continue

            if not os.path.isdir(rom_samples_solver_dir):
                print(f"Warning: samples_solver directory does not exist for {rom_name}.")
                print("Sample markers will be omitted for this case.")
                print(f"  Missing directory: {rom_samples_solver_dir}")
'''

if "samples_solver directory does not exist" not in text:
    text = replace_once(text, old, new, "samples_solver directory warning")
else:
    print("samples_solver missing-directory warning already present, skipping")

# ----------------------------------------------------------------------
# 12. Load sample cells inside each animation frame
# ----------------------------------------------------------------------
old = '''                        error_percent_norm = error_percent_history[plot_indx]

                        if plot_indx == 0:
'''

new = '''                        error_percent_norm = error_percent_history[plot_indx]

                        sample_cells = load_samples_solver_cells(
                            samples_solver_dir=rom_samples_solver_dir,
                            iter_num=actual_iter,
                            cell_num=cell_num
                        )

                        sample_x = x[sample_cells]
                        sample_y = rom_state_vector[sample_cells]

                        if plot_indx == 0:
'''

if "sample_cells = load_samples_solver_cells(" not in text:
    text = replace_once(text, old, new, "sample cell loading inside animation frame")
else:
    print("sample cell loading already present, skipping")

# ----------------------------------------------------------------------
# 13. Add red sampled-solver markers on first frame
# ----------------------------------------------------------------------
old = '''                            p2, = ax.plot(
                                x,
                                rom_state_vector,
                                ls='--',
                                c='tab:red',
                                lw=2,
                                label=f'{rom_legend}: {rom_name}'
                            )

                            ax.set_ylabel(y_label)
'''

new = '''                            p2, = ax.plot(
                                x,
                                rom_state_vector,
                                ls='--',
                                c='tab:red',
                                lw=2,
                                label=f'{rom_legend}: {rom_name}'
                            )

                            p3, = ax.plot(
                                sample_x,
                                sample_y,
                                marker='o',
                                ls='None',
                                c='red',
                                ms=5,
                                label='sampled solver cells'
                            )

                            ax.set_ylabel(y_label)
'''

if "label='sampled solver cells'" not in text:
    text = replace_once(text, old, new, "sample marker plot")
else:
    print("sample marker plot already present, skipping")

# ----------------------------------------------------------------------
# 14. Update marker data on later frames
# ----------------------------------------------------------------------
old = '''                            p1.set_ydata(fom_state_vector)
                            p2.set_ydata(rom_state_vector)

                            text_box.set_text(
'''

new = '''                            p1.set_ydata(fom_state_vector)
                            p2.set_ydata(rom_state_vector)
                            p3.set_data(sample_x, sample_y)

                            text_box.set_text(
'''

if "p3.set_data(sample_x, sample_y)" not in text:
    text = replace_once(text, old, new, "sample marker update")
else:
    print("sample marker update already present, skipping")

target_file.write_text(text)
print(f"Patched {target_file}")
