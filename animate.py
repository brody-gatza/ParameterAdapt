from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # HPC-safe non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


def read_prim_data(args):
    iter_num, result_dir = args
    filename = os.path.join(result_dir, f"{iter_num}iteration_prim.npy")
    return np.load(filename)


def get_existing_iter_list(result_dir, requested_iter_list, case_name="case"):
    """
    Returns only the iterations whose files actually exist.

    Expected file format:
        {iter_num}iteration_prim.npy
    """

    existing_iter_list = []
    missing_iter_list = []

    for iter_num in requested_iter_list:
        filename = os.path.join(result_dir, f"{iter_num}iteration_prim.npy")

        if os.path.exists(filename):
            existing_iter_list.append(iter_num)
        else:
            missing_iter_list.append(iter_num)

    existing_iter_list = np.array(existing_iter_list, dtype=int)
    missing_iter_list = np.array(missing_iter_list, dtype=int)

    print("------------------------------------------------------------")
    print(f"Checking files for {case_name}")
    print("Result directory:")
    print(f"  {result_dir}")
    print(f"Requested files: {len(requested_iter_list)}")
    print(f"Existing files:  {len(existing_iter_list)}")
    print(f"Missing files:   {len(missing_iter_list)}")

    if len(existing_iter_list) > 0:
        print(f"First existing iteration: {existing_iter_list[0]}")
        print(f"Last existing iteration:  {existing_iter_list[-1]}")

    if len(missing_iter_list) > 0:
        print(f"First missing iteration:  {missing_iter_list[0]}")
        print(f"Last missing iteration:   {missing_iter_list[-1]}")

    print("------------------------------------------------------------")

    return existing_iter_list


def gather_case_data(
    result_dir,
    requested_iter_list,
    num_processes,
    case_name,
    save_gathered_data=False
):
    """
    Gathers all existing iteration files for one case.

    Optional saves:
        prim_gathered.npy
        iter_list_gathered.npy

    Returns:
        full_prim
        existing_iter_list
        success

    full_prim shape:
        variable, cell, time
    """

    existing_iter_list = get_existing_iter_list(
        result_dir=result_dir,
        requested_iter_list=requested_iter_list,
        case_name=case_name
    )

    if len(existing_iter_list) == 0:
        print(f"Warning: no existing iteration files found for {case_name}. Skipping gather.")
        return None, None, False

    args_list = [(it, result_dir) for it in existing_iter_list]

    with Pool(processes=num_processes) as pool:
        prim_results = list(
            tqdm(
                pool.imap(read_prim_data, args_list),
                total=len(args_list),
                desc=case_name
            )
        )

    full_prim = np.array(prim_results)

    # Original shape after loading:
    #     time, variable, cell
    #
    # Desired shape:
    #     variable, cell, time
    full_prim = np.transpose(full_prim, (1, 2, 0))

    print(f"Finished gathering data for {case_name}.")
    print(f"Gathered data shape for {case_name}: {full_prim.shape}")

    if save_gathered_data:
        gathered_filename = os.path.join(result_dir, 'prim_gathered.npy')
        iter_list_filename = os.path.join(result_dir, 'iter_list_gathered.npy')

        np.save(gathered_filename, full_prim)
        np.save(iter_list_filename, existing_iter_list)

        print(f"Saved gathered data for {case_name} to:")
        print(f"  {gathered_filename}")
        print(f"Saved gathered iteration list for {case_name} to:")
        print(f"  {iter_list_filename}")
    else:
        print(f"Gathered data for {case_name} was kept in memory only.")
        print("No gathered .npy files were saved.")

    return full_prim, existing_iter_list, True


def load_gathered_case_data(result_dir, case_name):
    """
    Loads previously saved gathered data files.

    Expected files:
        prim_gathered.npy
        iter_list_gathered.npy
    """

    gathered_filename = os.path.join(result_dir, 'prim_gathered.npy')
    iter_list_filename = os.path.join(result_dir, 'iter_list_gathered.npy')

    if not os.path.exists(gathered_filename):
        print(f"Warning: gathered data file not found for {case_name}.")
        print(f"  Missing file: {gathered_filename}")
        return None, None, False

    if not os.path.exists(iter_list_filename):
        print(f"Warning: gathered iteration list file not found for {case_name}.")
        print(f"  Missing file: {iter_list_filename}")
        return None, None, False

    print(f"Loading gathered data for {case_name}:")
    print(f"  {gathered_filename}")
    full_prim = np.load(gathered_filename)

    print(f"Loading gathered iteration list for {case_name}:")
    print(f"  {iter_list_filename}")
    iter_list = np.load(iter_list_filename)

    print(f"Loaded gathered data for {case_name}.")
    print(f"Gathered data shape for {case_name}: {full_prim.shape}")
    print(f"Number of gathered iterations for {case_name}: {len(iter_list)}")

    return full_prim, iter_list, True


def calculate_percent_error_norm(
    fom_state_vector,
    rom_state_vector,
    fom_zero_tol=1.0e-12
):
    """
    Calculates the percent relative error vector cell-by-cell, then returns
    the norm of that percent error vector.
    """

    fom_safe = np.where(
        np.abs(fom_state_vector) > fom_zero_tol,
        fom_state_vector,
        np.nan
    )

    percent_error_vector = ((fom_state_vector - rom_state_vector) / fom_safe) * 100.0

    percent_error_vector = np.nan_to_num(
        percent_error_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    error_percent_norm = np.linalg.norm(percent_error_vector)

    return error_percent_norm, percent_error_vector


def get_fom_y_limits(
    fom_data,
    var_to_plot,
    fom_time_indices,
    padding_fraction=0.05,
    minimum_padding=1.0e-12
):
    """
    Determines y-axis limits from FOM data only.

    Uses:
        fom_data[var_to_plot, :, fom_time_indices]

    Returns:
        y_min
        y_max
    """

    fom_slice = fom_data[var_to_plot, :, fom_time_indices]

    finite_values = fom_slice[np.isfinite(fom_slice)]

    if finite_values.size == 0:
        print("Warning: no finite FOM values found for y-limit calculation.")
        print("Using fallback y-limits [-1, 1].")
        return -1.0, 1.0

    fom_min = np.min(finite_values)
    fom_max = np.max(finite_values)

    data_range = fom_max - fom_min

    if data_range <= 0.0:
        padding = max(abs(fom_max) * padding_fraction, minimum_padding)
    else:
        padding = data_range * padding_fraction

    y_min = fom_min - padding
    y_max = fom_max + padding

    return y_min, y_max


if __name__ == '__main__':

    # ============================================================
    # Inputs
    # ============================================================

    directory = '/kuhpc/scratch/capl/b323g408/1D_RDE_matrix/'

    # Single FOM path
    fom_result_dir = '/kuhpc/scratch/capl/b323g408/1D_RDE/old_results/FOM_results/'

    # ------------------------------------------------------------
    # ROM case list settings
    # ------------------------------------------------------------

    s_list = [1, 2, 50, 10, 15, 20]
    u_list = [1, 2, 5, 10, 15, 20]

    rom_label = os.path.join('AROM_results', 'cons_prim')

    animation_output_subdir = os.path.join('AROM_results', 'error', 'figures')

    rom_names = []
    rom_case_dirs = []
    rom_result_dirs = []
    rom_animation_base_dirs = []

    for s_num in s_list:
        for u_num in u_list:

            rom_name = f's{s_num}_u{u_num}'

            rom_case_dir = os.path.join(
                directory,
                rom_name
            )

            rom_result_dir = os.path.join(
                rom_case_dir,
                rom_label
            )

            rom_animation_base_dir = os.path.join(
                rom_case_dir,
                animation_output_subdir
            )

            rom_names.append(rom_name)
            rom_case_dirs.append(rom_case_dir)
            rom_result_dirs.append(rom_result_dir)
            rom_animation_base_dirs.append(rom_animation_base_dir)

    # ============================================================
    # Iteration settings
    # ============================================================

    start_iter = 0
    step_iter  = 100
    end_iter   = 200000

    step_plot  = 1

    # ============================================================
    # Mesh settings
    # ============================================================

    x_0      = 0
    x_f      = 0.288
    cell_num = 2000

    x_label = 'x [m]'

    # ------------------------------------------------------------
    # Variables to plot
    #
    # y-limits are now automatically determined from FOM data.
    # No user-input y_min or y_max values are needed.
    # ------------------------------------------------------------

    variables_to_plot = [
        {
            'var_index': 0,
            'var_name': 'density',
            'y_label': r'$\rho$'
        },
        {
            'var_index': 1,
            'var_name': 'velocity',
            'y_label': 'u [m/s]'
        },
        {
            'var_index': 2,
            'var_name': 'pressure',
            'y_label': 'P [Pa]'
        },
        {
            'var_index': 3,
            'var_name': 'temperature',
            'y_label': 'T [K]'
        },
        {
            'var_index': 4,
            'var_name': 'reactant',
            'y_label': 'Y reactant'
        },
        {
            'var_index': 5,
            'var_name': 'hrr',
            'y_label': 'heat release [W]'
        }
    ]

    fom_legend = 'FOM'
    rom_legend = 'ROM'

    num_processes = 1

    # ============================================================
    # Output/control switches
    # ============================================================

    gather_data = True

    save_gathered_data = True

    save_animation = True

    fps_input = 30

    frame_hold = 1

    skip_existing_animation = False

    # Y-axis padding based on FOM data range.
    #
    # Example:
    #     y_padding_fraction = 0.05 gives 5 percent padding above and below.
    y_padding_fraction = 0.05

    fom_zero_tol = 1.0e-12

    requested_iter_list = np.arange(start_iter, end_iter, step_iter, dtype=int)

    # ============================================================
    # Gather or load FOM data
    # ============================================================

    if gather_data:

        print('Gathering FOM data ...')
        print('FOM data directory:')
        print(f'  {fom_result_dir}')

        fom_data, fom_iter_list, fom_success = gather_case_data(
            result_dir=fom_result_dir,
            requested_iter_list=requested_iter_list,
            num_processes=num_processes,
            case_name='FOM',
            save_gathered_data=save_gathered_data
        )

    else:

        print('Loading previously gathered FOM data ...')
        print('FOM data directory:')
        print(f'  {fom_result_dir}')

        fom_data, fom_iter_list, fom_success = load_gathered_case_data(
            result_dir=fom_result_dir,
            case_name='FOM'
        )

    if not fom_success:
        raise RuntimeError(
            "No FOM data were available. Cannot continue."
        )

    # ============================================================
    # Generate animations
    # ============================================================

    if save_animation:

        print('Generating animations ...')

        x = np.linspace(x_0, x_f, cell_num)

        for rom_name, rom_result_dir, rom_animation_base_dir in zip(
            rom_names,
            rom_result_dirs,
            rom_animation_base_dirs
        ):

            print("============================================================")
            print(f'Processing ROM case {rom_name}')

            if not os.path.isdir(rom_result_dir):
                print(f"Warning: ROM directory does not exist. Skipping {rom_name}.")
                print(f"  Missing directory: {rom_result_dir}")
                continue

            # ----------------------------------------------------
            # Gather or load this ROM case
            # ----------------------------------------------------

            if gather_data:

                print(f'Gathering ROM data for {rom_name} ...')
                print('ROM data directory:')
                print(f'  {rom_result_dir}')

                rom_data, rom_iter_list, rom_success = gather_case_data(
                    result_dir=rom_result_dir,
                    requested_iter_list=requested_iter_list,
                    num_processes=num_processes,
                    case_name=rom_name,
                    save_gathered_data=save_gathered_data
                )

            else:

                print(f'Loading previously gathered ROM data for {rom_name} ...')
                print('ROM data directory:')
                print(f'  {rom_result_dir}')

                rom_data, rom_iter_list, rom_success = load_gathered_case_data(
                    result_dir=rom_result_dir,
                    case_name=rom_name
                )

            if not rom_success:
                print(f"Warning: could not get ROM data for {rom_name}. Skipping.")
                continue

            # ----------------------------------------------------
            # Find common iterations between FOM and this ROM case
            # ----------------------------------------------------

            common_iter_list = np.intersect1d(fom_iter_list, rom_iter_list)

            if len(common_iter_list) == 0:
                print(f"Warning: no common iterations between FOM and {rom_name}. Skipping.")
                continue

            common_iter_list = common_iter_list[::step_plot]

            if len(common_iter_list) == 0:
                print(f"Warning: no common iterations after applying step_plot for {rom_name}. Skipping.")
                continue

            estimated_duration = len(common_iter_list) * frame_hold / fps_input

            print(f"Common iterations for {rom_name}: {len(common_iter_list)}")
            print(f"First common iteration: {common_iter_list[0]}")
            print(f"Last common iteration:  {common_iter_list[-1]}")
            print(f"GIF fps: {fps_input}")
            print(f"Frame hold: {frame_hold}")
            print(f"Estimated GIF duration: {estimated_duration:.2f} seconds")

            fom_iter_to_index = {
                iter_num: index for index, iter_num in enumerate(fom_iter_list)
            }

            rom_iter_to_index = {
                iter_num: index for index, iter_num in enumerate(rom_iter_list)
            }

            # FOM time indices for the common iterations.
            # These are used to determine plot limits from FOM data only.
            fom_common_time_indices = np.array(
                [fom_iter_to_index[actual_iter] for actual_iter in common_iter_list],
                dtype=int
            )

            # ====================================================
            # Loop over variables
            # ====================================================

            for var_config in variables_to_plot:

                var_to_plot = var_config['var_index']
                var_name    = var_config['var_name']
                y_label     = var_config['y_label']

                print("------------------------------------------------------------")
                print(f"Generating animation for {rom_name}, variable {var_name}")
                print(f"Variable index: {var_to_plot}")

                # ------------------------------------------------
                # Determine y-axis limits from FOM data only
                # ------------------------------------------------

                y_min, y_max = get_fom_y_limits(
                    fom_data=fom_data,
                    var_to_plot=var_to_plot,
                    fom_time_indices=fom_common_time_indices,
                    padding_fraction=y_padding_fraction
                )

                print("Y-axis limits determined from FOM data:")
                print(f"  y_min = {y_min:.6e}")
                print(f"  y_max = {y_max:.6e}")

                # ------------------------------------------------
                # Calculate percent relative error magnitude
                # ------------------------------------------------

                error_percent_history = []

                for actual_iter in common_iter_list:

                    fom_data_indx = fom_iter_to_index[actual_iter]
                    rom_data_indx = rom_iter_to_index[actual_iter]

                    fom_state_vector = fom_data[var_to_plot, :, fom_data_indx]
                    rom_state_vector = rom_data[var_to_plot, :, rom_data_indx]

                    error_percent_norm, _ = calculate_percent_error_norm(
                        fom_state_vector=fom_state_vector,
                        rom_state_vector=rom_state_vector,
                        fom_zero_tol=fom_zero_tol
                    )

                    error_percent_history.append(error_percent_norm)

                error_percent_history = np.array(error_percent_history)

                # ------------------------------------------------
                # Make animation
                # ------------------------------------------------

                fig, ax = plt.subplots(1, 1)

                writer = PillowWriter(
                    fps=fps_input,
                    metadata=dict(artist='Ali')
                )

                animation_output_dir = rom_animation_base_dir

                os.makedirs(animation_output_dir, exist_ok=True)

                animation_filename = os.path.join(
                    animation_output_dir,
                    f'{rom_name}_{var_name}.gif'
)

                if skip_existing_animation and os.path.exists(animation_filename):
                    print("Animation already exists. Skipping to avoid overwrite:")
                    print(f"  {animation_filename}")
                    plt.close(fig)
                    continue

                print("Animation output directory:")
                print(f"  {animation_output_dir}")
                print("Final animation file:")
                print(f"  {animation_filename}")

                with writer.saving(fig, animation_filename, dpi=100):

                    for plot_indx, actual_iter in enumerate(common_iter_list):

                        fom_data_indx = fom_iter_to_index[actual_iter]
                        rom_data_indx = rom_iter_to_index[actual_iter]

                        fom_state_vector = fom_data[var_to_plot, :, fom_data_indx]
                        rom_state_vector = rom_data[var_to_plot, :, rom_data_indx]

                        error_percent_norm = error_percent_history[plot_indx]

                        if plot_indx == 0:

                            p1, = ax.plot(
                                x,
                                fom_state_vector,
                                ls='-',
                                c='black',
                                lw=4,
                                label=fom_legend
                            )

                            p2, = ax.plot(
                                x,
                                rom_state_vector,
                                ls='--',
                                c='tab:red',
                                lw=2,
                                label=f'{rom_legend}: {rom_name}'
                            )

                            ax.set_ylabel(y_label)
                            ax.set_xlabel(x_label)

                            # Limits are determined from FOM data only.
                            ax.set_ylim([y_min, y_max])

                            ax.legend(
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
                                    f'Case: {rom_name}\n'
                                    f'Variable: {var_name}\n'
                                    f'Iteration: {actual_iter}\n'
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

                        else:

                            p1.set_ydata(fom_state_vector)
                            p2.set_ydata(rom_state_vector)

                            text_box.set_text(
                                (
                                    f'Case: {rom_name}\n'
                                    f'Variable: {var_name}\n'
                                    f'Iteration: {actual_iter}\n'
                                    f'Percent error norm: {error_percent_norm:.6e} %'
                                )
                            )

                        plt.tight_layout()

                        for _ in range(frame_hold):
                            writer.grab_frame()

                plt.close(fig)

                print(f'Animation saved as {animation_filename}')

            del rom_data
            del rom_iter_list

        del fom_data
        del fom_iter_list