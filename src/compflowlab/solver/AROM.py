import os
import numpy as np


from compflowlab.utils import reshape_func
from compflowlab.boundary_condition import bc_func
from compflowlab.rom.basis_func import adapt_basis
from compflowlab.rom.sampling_func import hyper_precompute

from ..utils.classes import MovingAverage
import copy

def snapshot_state(state):
    """
    Create an independent snapshot of the complete mutable solver state.

    Open file handles are preserved by reference because they cannot and
    should not be deep-copied.
    """
    reference_only_keys = {
        "error_output_files",
        "conservation_error_output_files",
    }

    # Preserve external resources by reference.
    preserved_references = {
        key: state[key]
        for key in reference_only_keys
        if key in state
    }

    # Deep-copy everything else as one object. Copying the dictionary as one
    # object preserves any shared references between its copied entries.
    copyable_state = {
        key: value
        for key, value in state.items()
        if key not in reference_only_keys
    }

    snapshot = copy.deepcopy(copyable_state)

    # Reattach the original external resources.
    snapshot.update(preserved_references)

    return snapshot

def slope_weight(n, nm1, slope, ):
    tol = 5 # 10
    thresh = 50
    weight = np.where(slope > 0, n/nm1,
        np.where(slope < 0, nm1/n, 0.0)
    )

    # I would like to negate, but it wont work like this bc the sign is applied outside this function
    # weight = np.where(np.abs(weight) < tol, -1, np.where(np.abs(weight) > thresh, 2, 1))
    weight = np.where(np.abs(weight) < tol, 0, np.where(np.abs(weight) > thresh, 2, 1))


    # weight = np.ones_like(slope)

    return weight

def _format_array_line(iter, arr):
    arr = np.atleast_1d(arr)
    return str(iter) + "," + ",".join(f"{x:.17e}" for x in arr) + "\n"



def _get_error_output_base_names():
    """
    Base error-output names.

    Every base name gets:
        raw output
        _short_ma
        _long_ma
        _ma_counter
        _slope_counter
        _slope_ratio
    """
    return [
        "prim_interp_max",
        "prim_interp_avg",
        "prim_proj_max",
        "prim_proj_avg",

        "cons_interp_max",
        "cons_interp_avg",
        "cons_proj_max",
        "cons_proj_avg",
    ]


def _get_error_output_file_names(include_sampling_freq=True):
    """
    Returns the complete list of error-output file keys.

    This is the single source of truth used by:
        _open_error_output_files(...)
        _write_error_output_array_map(...)
    """
    base_names = _get_error_output_base_names()

    file_names = []

    file_names.extend(base_names)

    for suffix in [
        "_short_ma",
        "_long_ma",
        "_ma_counter",
        "_ma_counter_sum",
    ]:
        for name in base_names:
            file_names.append(f"{name}{suffix}")

    for suffix in [
        "_slope_counter",
        "_slope_ratio",
    ]:
        for name in base_names:
            file_names.append(f"{name}{suffix}")

    if include_sampling_freq:
        file_names.append("sampling_freq")

    return file_names


def _write_error_output_array_map(state, iter, output_array_map):
    """
    Write all array-based error outputs using the centralized file list.

    sampling_freq is intentionally excluded because it is scalar-formatted
    separately.
    """
    for name in _get_error_output_file_names(include_sampling_freq=False):
        if name not in output_array_map:
            raise KeyError(
                f"Missing output array for '{name}'. "
                "Check _get_error_output_file_names() and output_array_map."
            )

        _write_error_output_line(
            state,
            name,
            _format_array_line(iter, output_array_map[name])
        )


def _open_error_output_files(state, dir_results, mode):
    """
    Opens error-output files once and stores file handles in state.

    mode should be:
        "w" on the first error-output iteration
        "a" if restarting/appending
    """

    file_names = _get_error_output_file_names()

    state["error_output_files"] = {}

    for name in file_names:
        file_path = os.path.join(dir_results, name + ".txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        state["error_output_files"][name] = open(
            file_path,
            mode,
            buffering=1024 * 1024
        )

    state["error_output_flush_counter"] = 0


def _write_error_output_line(state, name, line):
    state["error_output_files"][name].write(line)


def _open_conservation_error_output_files(state, dir_results, mode):
    # These files are independent of the periodically evaluated ROM errors.
    os.makedirs(dir_results, exist_ok=True)
    state["conservation_error_output_files"] = {}

    for name in ["error_cons", "error_cons_perct"]:
        file_path = os.path.join(dir_results, name + ".txt")
        state["conservation_error_output_files"][name] = open(
            file_path,
            mode,
            buffering=1024 * 1024,
        )

    state["conservation_error_output_flush_counter"] = 0


def _write_conservation_error_output_line(state, name, line):
    state["conservation_error_output_files"][name].write(line)


def _flush_conservation_error_output_files(state, force=False):
    if "conservation_error_output_files" not in state:
        return

    flush_interval = state.get("error_output_flush_interval", 100)
    state["conservation_error_output_flush_counter"] += 1

    if force or state["conservation_error_output_flush_counter"] >= flush_interval:
        for file in state["conservation_error_output_files"].values():
            file.flush()

        state["conservation_error_output_flush_counter"] = 0


def close_conservation_error_output_files(state):
    if "conservation_error_output_files" not in state:
        return

    for file in state["conservation_error_output_files"].values():
        try:
            file.flush()
            file.close()
        except Exception:
            pass

    del state["conservation_error_output_files"]
    state.pop("conservation_error_output_flush_counter", None)


def _flush_error_output_files(state, force=False):
    if "error_output_files" not in state:
        return

    flush_interval = state.get("error_output_flush_interval", 100)

    state["error_output_flush_counter"] += 1

    if force or state["error_output_flush_counter"] >= flush_interval:
        for file in state["error_output_files"].values():
            file.flush()

            # Uncomment this if you want stronger crash protection, but it is slower.
            # os.fsync(file.fileno())

        state["error_output_flush_counter"] = 0


def close_error_output_files(state):
    """
    Call this at the end of the run if possible.

    This is safe to call even if files were never opened.
    """

    if "error_output_files" not in state:
        return

    for file in state["error_output_files"].values():
        try:
            file.flush()
            file.close()
        except Exception:
            pass

    del state["error_output_files"]
    state.pop("error_output_flush_counter", None)

def precomputer(solver_param):

    print('Initializing Adaptive ROM')

    rom_param = {}

    training_data_cons = reshape_func.assemble_snapshots(solver_param)

    # number of snapshot
    num_snapshot = len(training_data_cons[0,0,:])

    # reference profile
    q_ref = training_data_cons[:,:,-1]

    # center data
    centered_data = training_data_cons - q_ref[:,:,np.newaxis]

    # normalizing factors
    l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
    norm_factor        = np.mean(l2_factors, axis=1)

    # centered_normalized data
    cen_norm_data = centered_data / norm_factor[:, np.newaxis, np.newaxis]

    # data matrix
    tall_thin_data = cen_norm_data.reshape(-1, num_snapshot)

    # perform SVD
    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

    # finalize the basis
    basis = V[:,0:-1]

    # wrap up and exit the function
    denormalizor = np.repeat(norm_factor, solver_param['cell_number'])
    normalizor   = 1/denormalizor

    rom_param['basis']                = basis
    rom_param['q_ref']                = q_ref.ravel()
    rom_param['norm']                 = normalizor
    rom_param['denorm']               = denormalizor
    rom_param['F']                    = tall_thin_data
    rom_param['Q_R']                  = rom_param['basis'].T @ rom_param['F']
    rom_param['qr0']                  = basis.T @ tall_thin_data[:,0]

    return rom_param

def prepare_to_store_FOM(solver_param,state,rom_param):

    # prepare data to save
    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]
    state['res_save']          = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['d_flux_dx'])[:,2:-2]

    if solver_param['gas_model'] == 'Air':

        state['prim_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

    return state

def prepare_to_store_ROM(solver_param,state,rom_param):

    # prepare data to save
    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]
    state['res_save']          = np.zeros_like(state['cons_results_save'].ravel())-1
    state['res_save'][rom_param['S_indx_solver']] = state['d_flux_dx']
    state['res_save']          = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number']-4,state['res_save'])

    if solver_param['gas_model'] == 'Air':

        state['prim_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

    return state

def results_recorder_FOM(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

def results_recorder_FOM_error(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param["dir_results"], "error")
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'FOM_data' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'FOM_data' ,f"{save_title}_prim.npy"), state['prim_results_save'])

def results_recorder_trans(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

    # Save rom related parameters
    np.save(os.path.join(dir_results,'basis'         ,f"{save_title}_basis.npy")          , rom_param['basis'])
    np.save(os.path.join(dir_results,'samples_user'  ,f"{save_title}_samples_user.npy")   , rom_param['S_indx_user'])
    np.save(os.path.join(dir_results,'samples_solver',f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])
    np.save(os.path.join(dir_results,'q_ref'         ,f"{save_title}_q_ref.npy")          , rom_param['q_ref'])
    np.save(os.path.join(dir_results,'norm'          ,f"{save_title}_norm.npy")           , rom_param['norm'])
    np.save(os.path.join(dir_results,'denorm'        ,f"{save_title}_denorm.npy")         , rom_param['denorm'])

def results_recorder_ROM(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

    # Save rom related parameters
    np.save(os.path.join(dir_results,'basis'         ,f"{save_title}_basis.npy")          , rom_param['basis'])
    np.save(os.path.join(dir_results,'samples_user'  ,f"{save_title}_samples_user.npy")   , rom_param['S_indx_user'])
    np.save(os.path.join(dir_results,'samples_solver',f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

def update_sampling_points(solver_param,state,physics,time_integration,rom_param,sampling_adapt_freq):
    # Compute a time step in the future if using future FGS
    if solver_param['sampling_method'] == 'FFGS' or solver_param['sampling_method'] == 'FFGSR':

        # Save the state to restore to
        restore_solver = snapshot_state(solver_param)

        # Load the previous future state and export as the field solution
        if ( not ( solver_param['iter'] == int(solver_param['FOM2ROM_trans_iter'])) ):
            if solver_param['sampling_method'] == 'FFGS':
                # Load the stored future state
                state = snapshot_state(rom_param['future_state'])

            elif solver_param['sampling_method'] == 'FFGSR':
                # Evaluate the full field
                solver_param['hyper'] = False

                # take FOM step for initial training
                state = physics.residual_calculator(solver_param,rom_param,state)
                state = time_integration.advance_time(solver_param,rom_param,state,physics)

                # post process part
                if solver_param['injection']:

                    state = physics.injection_correction(solver_param,state)

            Q_bar_star_new = state['Q_cons']
            Q_bar_star_new_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_star_new)

            # adapt basis with newly found sanpshot
            rom_param = adapt_basis(solver_param,rom_param,Q_bar_star_new_solver_int)

            # find corrected qr (projected with new basis)
            new_qr = np.transpose(rom_param['basis']) @ rom_param['F'][:,-1]

            # update states
            corrected_cent_norm = rom_param['basis'] @ new_qr
            rom_param['F'][:,-1]   = corrected_cent_norm
            rom_param['Q_R'][:,-1] = new_qr

            # update prim state
            state = physics.cons2prim_converter(solver_param,state)

            # update the ghost cells
            state = bc_func.update_ghost_cell(solver_param,state)

            # update prim state
            state = physics.prim2cons_converter(solver_param,state)

            # prepare results to save
            state = prepare_to_store_FOM(solver_param,state,rom_param)

            # save the data
            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_ROM(solver_param,state,rom_param)

        # Save the state to restore to
        restore_state = snapshot_state(state)

        # adjust the solver parameters for taking large time step FOM
        solver_param['hyper'] = False
        solver_param['dt']    = sampling_adapt_freq * solver_param['dt']

        # take one FOM step
        state = physics.residual_calculator(solver_param,rom_param,state)
        state = time_integration.advance_time(solver_param,rom_param,state,physics)

        if solver_param['injection']:

            state = physics.injection_correction(solver_param,state)

        Q_bar_star_new = state['Q_cons']
        Q_bar_star_new_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_star_new)

        # some sampling methods (ex. FGS) need this for sampling
        rom_param['Q_bar'] = Q_bar_star_new.copy()

        # reset solver parameters to samller time step setup (user defined setup)
        # Q_bar_new_solver_int    = Q_bar_star_new_solver_int
        solver_param['hyper']   = True
        solver_param['dt']      = solver_param['dt'] / sampling_adapt_freq

        # Disabling basis update based on prediction
        # # adapt basis with newly found sanpshot
        # rom_param = adapt_basis(solver_param,rom_param,Q_bar_new_solver_int)

        # # find corrected qr (projected with new basis)
        # new_qr = np.transpose(rom_param['basis']) @ rom_param['F'][:,-1]

        # # update states
        # corrected_cent_norm = rom_param['basis'] @ new_qr
        # rom_param['F'][:,-1]   = corrected_cent_norm
        # rom_param['Q_R'][:,-1] = new_qr

        # Not sure if this is needed, we are already computing the true solution with FOM (Maybe useful for future prediction but not for regular time step)
        # # find new solution
        # Q_tilda_correct_solver_int = q_ref + (denormalizor * corrected_cent_norm)
        # Q_tilda_correct_solver_full= reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)
        # state['Q_cons'] = Q_tilda_correct_solver_full

        # Save the future predicted state for writing later
        if solver_param['sampling_method'] == 'FFGS':
            rom_param['future_state'] = snapshot_state(state)

        # Restore to the saved states and update only future values
        state.clear()
        state.update(restore_state)
        solver_param.clear()
        solver_param.update(restore_solver)

    # Otherwise, compute the next time step
    else:

        # adjust the solver parameters for regular time step FOM
        solver_param['hyper'] = False

        # take one FOM step
        state = physics.residual_calculator(solver_param,rom_param,state)
        state = time_integration.advance_time(solver_param,rom_param,state,physics)

        if solver_param['injection']:

            state = physics.injection_correction(solver_param,state)

        Q_bar_star_new = state['Q_cons'].copy()
        Q_bar_star_new_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_star_new)

        # some sampling methods (ex. FGS) need this for sampling
        rom_param['Q_bar'] = Q_bar_star_new.copy()

        # reset solver parameters to samller time step setup (user defined setup)
        Q_bar_new_solver_int    = Q_bar_star_new_solver_int
        solver_param['hyper']   = True

        # adapt basis with newly found sanpshot
        rom_param = adapt_basis(solver_param,rom_param,Q_bar_new_solver_int)

        # find corrected qr (projected with new basis)
        new_qr = np.transpose(rom_param['basis']) @ rom_param['F'][:,-1]

        # update states
        corrected_cent_norm = rom_param['basis'] @ new_qr
        rom_param['F'][:,-1]   = corrected_cent_norm
        rom_param['Q_R'][:,-1] = new_qr

        # BUG? Why is this called twice??
        # # post process part
        # if solver_param['injection']:

        #     state = physics.injection_correction(solver_param,state)

        # update prim state
        state = physics.cons2prim_converter(solver_param,state)

        # update the ghost cells
        state = bc_func.update_ghost_cell(solver_param,state)

        # update prim state
        state = physics.prim2cons_converter(solver_param,state)

        # prepare results to save
        state = prepare_to_store_FOM(solver_param,state,rom_param)

        # save the data
        if solver_param['iter'] % solver_param['save_interval'] == 0:

            results_recorder_ROM(solver_param,state,rom_param)

    # Update Samples
    rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

    return state, rom_param

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    #############################################
    # This function is taking one time step     #
    # using adaptive ROM algorithm. It will run #
    # FOM initially but then will turn into ROM #
    # with evolving basis and sample adaptation #
    #############################################

    iter = solver_param['iter']

    if solver_param['error_check']:
        state['Q_cons_pre'] = copy.deepcopy(state['Q_cons'])
        state['Q_cons_inj'] = np.zeros(solver_param['num_state_var'])

    if iter <= int(solver_param['FOM2ROM_trans_iter']):

        # take FOM step for initial training
        state = physics.residual_calculator(solver_param,rom_param,state)
        state = time_integration.advance_time(solver_param,rom_param,state,physics)

        # post process part
        if solver_param['injection']:

            state = physics.injection_correction(solver_param,state)

        # update prim state
        state = physics.cons2prim_converter(solver_param,state)

        # update the ghost cells
        state = bc_func.update_ghost_cell(solver_param,state)

        # update prim state
        state = physics.prim2cons_converter(solver_param,state)

        # prepare results to save
        state = prepare_to_store_FOM(solver_param,state,rom_param)

        if iter != int(solver_param['FOM2ROM_trans_iter']):

            # save solution
            results_recorder_FOM(solver_param,state,rom_param)

        elif iter == int(solver_param['FOM2ROM_trans_iter']):

            # adjust the training range
            solver_param['training_start_iter'] = int(iter-solver_param['init_training_win'])
            solver_param['training_end_iter'  ] = iter
            solver_param['training_step_iter' ] = 1
            solver_param['training_data_dir']   = os.path.join(solver_param['dir_results'], 'cons_prim')

            # build ROM
            rom_param = precomputer(solver_param)

            # create a full-state copy
            rom_param['Q_bar'] = state['Q_cons'].copy()

            # adjust sampling configuration
            solver_param['hyper'] = True

            # manually specify resampling time steps
            if solver_param['multi_samp']:

                block_size = 5000

                for i in range(2, 50):
                    if i == 2:
                        block_start = solver_param['FOM2ROM_trans_iter']
                    else:
                        block_start = actual_block_end
                    nominal_block_end = (i - 1) * block_size

                    # Move nominal_block_end forward until it lands on the sampling schedule
                    remainder = (nominal_block_end - block_start) % i

                    if remainder == 0:
                        actual_block_end = nominal_block_end
                    else:
                        actual_block_end = nominal_block_end + (i - remainder)

                    # Optional but recommended: do not go past num_step
                    actual_block_end = min(actual_block_end, solver_param['num_step'])

                    sampling_list = np.arange(
                        block_start,
                        actual_block_end,
                        i,
                        dtype=int
                    )

                    if i == 2:
                        solver_param['resample_iter_list'] = sampling_list
                    else:
                        solver_param['resample_iter_list'] = np.append(
                            solver_param['resample_iter_list'],
                            sampling_list
                        )

                    if actual_block_end >= solver_param['num_step']:
                        break

                solver_param['resample_iter_list'] = np.unique(solver_param['resample_iter_list'])
                sampling_adapt_freq = 2

            elif solver_param['force_FOM']:
                start_FOM = 86000
                end_FOM =   96000
                solver_param['resample_iter_list'] = np.arange(solver_param['FOM2ROM_trans_iter'],
                                                               start_FOM,
                                                               solver_param['unsampled_update_freq'],dtype=int)
                sampling_adapt_freq = solver_param['unsampled_update_freq']

                solver_param['resample_iter_list'] = np.append(solver_param['resample_iter_list'],np.arange(start_FOM,
                                                               end_FOM,
                                                               1,dtype=int))
                sampling_adapt_freq = solver_param['unsampled_update_freq']

                solver_param['resample_iter_list'] = np.append(solver_param['resample_iter_list'],np.arange(end_FOM,
                                                               solver_param['num_step'],
                                                               solver_param['unsampled_update_freq'],dtype=int))
                sampling_adapt_freq = solver_param['unsampled_update_freq']


            else:
                solver_param['resample_iter_list'] = np.arange(solver_param['FOM2ROM_trans_iter'],
                                                               solver_param['num_step'],
                                                               solver_param['unsampled_update_freq'],dtype=int)
                sampling_adapt_freq = solver_param['unsampled_update_freq']

            # find initial samples
            if solver_param['sampling_method'] == 'FFGS' or solver_param['sampling_method'] == 'FFGSR':
                state, rom_param = update_sampling_points(solver_param,state,physics,time_integration,rom_param,sampling_adapt_freq)
            else:
                rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

            # save ROM related param
            results_recorder_trans(solver_param,state,rom_param)

    else:
        # read basic parameters
        q_ref                  = rom_param['q_ref']
        normalizor             = rom_param['norm']
        denormalizor           = rom_param['denorm']

        # Q tilda (ROM) before any update
        Q_tilda_old            = state['Q_cons'].copy()
        Q_tilda_old_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_old)

        # This routine evalutes the FOM based on the current time step to calculate ROM errors
        if ( solver_param['error_check'] and not ( np.any(solver_param['iter'] == solver_param['resample_iter_list']) ) ):
        # if ( solver_param['error_check']):

            # Save the ROM state to restore to
            restore_state = snapshot_state(state)
            restore_solver = snapshot_state(solver_param)
            restore_rom = snapshot_state(rom_param)

            # Q_cons_save = state['Q_cons'].copy()
            # Q_prim_save = state['Q_prim'].copy()
            # hyper_save = solver_param['hyper']

            # disable hyper-reduction so the FOM is fully evaluated
            solver_param['hyper'] = False

            # take one FOM step
            state = physics.residual_calculator(solver_param,rom_param,state)
            state = time_integration.advance_time(solver_param,rom_param,state,physics)

            # post process part
            if solver_param['injection']:

                state = physics.injection_correction(solver_param,state)

            # Calculate the primitive variables for the FOM FOM solution
            # update prim state
            state = physics.cons2prim_converter(solver_param,state)

            # update the ghost cells
            state = bc_func.update_ghost_cell(solver_param,state)

            # update prim state
            state = physics.prim2cons_converter(solver_param,state)

            # prepare results to save
            state = prepare_to_store_FOM(solver_param,state,rom_param)

            # save the data
            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_FOM_error(solver_param,state,rom_param)

            # Store the FOM solution
            Q_cons_FOM = state['Q_cons'].copy()
            Q_prim_FOM = state['Q_prim'].copy()

            # Restore the states
            # state = restore_state
            # solver_param = restore_solver
            # rom_param = restore_rom
            state.clear()
            state.update(restore_state)
            solver_param.clear()
            solver_param.update(restore_solver)
            rom_param.clear()
            rom_param.update(restore_rom)
            # state['Q_cons'] = Q_cons_save.copy()
            # state['Q_prim'] = Q_prim_save.copy()
            # solver_param['hyper'] = hyper_save

        # Update the sampling frequency to reflect the current value if using multi_samp
        if solver_param['multi_samp'] or solver_param['force_FOM']:
            iter_ind = np.where(solver_param['resample_iter_list'] == iter)[0]
            if iter_ind.size > 0:
                sampling_adapt_freq = solver_param['resample_iter_list'][iter_ind + 1] - solver_param['resample_iter_list'][iter_ind]
                sampling_adapt_freq = int(sampling_adapt_freq[0])
                solver_param['unsampled_update_freq'] = sampling_adapt_freq
        else:
            sampling_adapt_freq = solver_param['unsampled_update_freq']

        # Compute FOM a time step to update the sampling points and ROM basis
        if np.any(solver_param['iter'] == solver_param['resample_iter_list']):
            state, rom_param = update_sampling_points(solver_param,state,physics,time_integration,rom_param,sampling_adapt_freq)

        # Run ROM at small time step (user defined dt)
        else:

            # find new solution only at sampled points
            state['Q_cons']    = Q_tilda_old
            state              = physics.residual_calculator(solver_param,rom_param,state)
            state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
            state              = time_integration.advance_time(solver_param,rom_param,state,physics)
            Q_bar_new_sampling = state['Q_cons']

            # Estimate full-state at unsampled points using old basis (DEIM Equation) -- PREDICTION STEP
            decen_norm_Q_bar_new_sampling           = normalizor[rom_param['S_indx_solver']]*(Q_bar_new_sampling-q_ref[rom_param['S_indx_solver']])
            C                                       = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]) @ decen_norm_Q_bar_new_sampling
            Q_bar_new_solver_int                    = q_ref + (denormalizor * (rom_param['basis'] @ C ))

            if solver_param['injection']:

                Q_bar_new_solver_full               = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                                                    solver_param['num_state_var'],
                                                                                    Q_bar_new_solver_int)

                state['Q_cons'] = Q_bar_new_solver_full
                state = physics.injection_correction(solver_param,state)
                Q_bar_new_solver_full = state['Q_cons']

                Q_bar_new_solver_int  = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                                            solver_param['num_state_var'],
                                                                            Q_bar_new_solver_full)

            # adapt basis with newly found sanpshot
            rom_param = adapt_basis(solver_param,rom_param,Q_bar_new_solver_int)

            # find corrected qr (projected with new basis)
            new_qr = np.transpose(rom_param['basis']) @ rom_param['F'][:,-1]

            # update states
            corrected_cent_norm    = rom_param['basis'] @ new_qr
            rom_param['F'][:,-1]   = corrected_cent_norm
            rom_param['Q_R'][:,-1] = new_qr

            # find new solution -- CORRECTION STEP
            Q_tilda_correct_solver_int= q_ref + (denormalizor * corrected_cent_norm)
            Q_tilda_correct_solver_full= reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)
            state['Q_cons'] = Q_tilda_correct_solver_full

            # post process part
            if solver_param['injection']:

                state = physics.injection_correction(solver_param,state)

            # update prim state
            state = physics.cons2prim_converter(solver_param,state)

            # update the ghost cells
            state = bc_func.update_ghost_cell(solver_param,state)

            # update prim state
            state = physics.prim2cons_converter(solver_param,state)

            # prepare results to save
            state = prepare_to_store_ROM(solver_param,state,rom_param)

            # save the data
            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_ROM(solver_param,state,rom_param)

        # Error checking routines
        if ( solver_param['error_check'] and not ( np.any(solver_param['iter'] == solver_param['resample_iter_list']) ) ):
        # if ( solver_param['error_check'] ):

            # Calculate the interpolation error
            # Not technically interpolation error, more like overall ROM error
            Q_cons_interp_error = np.abs(Q_cons_FOM - state['Q_cons'])
            Q_prim_interp_error = np.abs(Q_prim_FOM - state['Q_prim'])

            # Calculate the projection error
            Q_cons_FOM_int = normalizor * (reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_cons_FOM) - q_ref)
            Q_cons_FOM_int_proj = denormalizor * (rom_param['basis'] @ (rom_param['basis'].T @ Q_cons_FOM_int)) + q_ref
            Q_cons_FOM_proj = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_cons_FOM_int_proj)
            Q_cons_proj_error = np.abs(Q_cons_FOM - Q_cons_FOM_proj)

            # Save the ROM state to restore to
            restore_state = snapshot_state(state)
            restore_solver = snapshot_state(solver_param)

            # Convert the projected FOM to primitive variables
            state['Q_cons'] = Q_cons_FOM_proj

            # # post process part
            # if solver_param['injection']:

            #     state = physics.injection_correction(solver_param,state)

            # update prim state
            state = physics.cons2prim_converter(solver_param,state)

            # update the ghost cells
            state = bc_func.update_ghost_cell(solver_param,state)

            Q_prim_proj_error = np.abs(Q_prim_FOM - state['Q_prim'])

            # Restore the states
            # state = restore_state
            # solver_param = restore_solver
            state.clear()
            state.update(restore_state)
            solver_param.clear()
            solver_param.update(restore_solver)


            # Reshape the error vectors to extract specific variables
            Q_cons_interp_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons_interp_error)[:,2:-2]
            Q_cons_proj_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons_proj_error)[:,2:-2]
            Q_cons_FOM_max = np.max(reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons_FOM)[:,2:-2], axis=1)

            Q_prim_interp_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_interp_error)[:,2:-2]
            Q_prim_proj_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_proj_error)[:,2:-2]
            Q_prim_FOM_max = np.max(reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_FOM)[:,2:-2], axis=1)

            # Normalize the errors by the maximum value in the field
            Q_cons_interp_error_reshape = Q_cons_interp_error_reshape / Q_cons_FOM_max[:, np.newaxis]
            Q_cons_proj_error_reshape = Q_cons_proj_error_reshape / Q_cons_FOM_max[:, np.newaxis]

            Q_prim_interp_error_reshape = Q_prim_interp_error_reshape / Q_prim_FOM_max[:, np.newaxis]
            Q_prim_proj_error_reshape = Q_prim_proj_error_reshape / Q_prim_FOM_max[:, np.newaxis]

            # Calculate QoIs per variable
            cons_interp_max = np.max(Q_cons_interp_error_reshape, axis=1)
            # cons_interp_min = np.min(Q_cons_interp_error_reshape, axis=1)
            cons_interp_avg = np.mean(Q_cons_interp_error_reshape, axis=1)
            cons_proj_max = np.max(Q_cons_proj_error_reshape, axis=1)
            # cons_proj_min = np.min(Q_cons_proj_error_reshape, axis=1)
            cons_proj_avg = np.mean(Q_cons_proj_error_reshape, axis=1)

            prim_interp_max = np.max(Q_prim_interp_error_reshape, axis=1)
            # prim_interp_min = np.min(Q_prim_interp_error_reshape, axis=1)
            prim_interp_avg = np.mean(Q_prim_interp_error_reshape, axis=1)
            prim_proj_max = np.max(Q_prim_proj_error_reshape, axis=1)
            # prim_proj_min = np.min(Q_prim_proj_error_reshape, axis=1)
            prim_proj_avg = np.mean(Q_prim_proj_error_reshape, axis=1)

            # Write the error values
            dir_results = os.path.join(solver_param["dir_results"], "error")
            if iter == int(solver_param['FOM2ROM_trans_iter']) + 1:
                # Creates new files or clears existing files
                mode = "w"

                rom_param["error_output_flush_interval"] = solver_param.get("error_output_flush_interval", 100)

                if "error_output_files" not in rom_param:
                    _open_error_output_files(rom_param, dir_results, mode)

                rom_param['cons_interp_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
                rom_param['cons_interp_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])
                rom_param['cons_proj_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
                rom_param['cons_proj_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])

                rom_param['prim_interp_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                rom_param['prim_interp_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                rom_param['prim_proj_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                rom_param['prim_proj_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])

                cons_interp_max_slope_ratio = np.zeros_like(cons_interp_max)
                cons_interp_avg_slope_ratio = np.zeros_like(cons_interp_avg)
                cons_proj_max_slope_ratio   = np.zeros_like(cons_proj_max)
                cons_proj_avg_slope_ratio   = np.zeros_like(cons_proj_avg)

                prim_interp_max_slope_ratio = np.zeros_like(prim_interp_max)
                prim_interp_avg_slope_ratio = np.zeros_like(prim_interp_avg)
                prim_proj_max_slope_ratio   = np.zeros_like(prim_proj_max)
                prim_proj_avg_slope_ratio   = np.zeros_like(prim_proj_avg)

                # HARDCODED FOR NOW
                short_ma_window = 25
                long_ma_window = 100

                rom_param['cons_interp_max_short_ma']   = MovingAverage(short_ma_window, solver_param['num_state_var'])
                rom_param['cons_interp_max_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                rom_param['cons_interp_max_ma_counter'] = np.zeros(solver_param['num_state_var'])

                rom_param['cons_interp_avg_short_ma']   = MovingAverage(short_ma_window, solver_param['num_state_var'])
                rom_param['cons_interp_avg_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                rom_param['cons_interp_avg_ma_counter'] = np.zeros(solver_param['num_state_var'])

                rom_param['cons_proj_max_short_ma']     = MovingAverage(short_ma_window, solver_param['num_state_var'])
                rom_param['cons_proj_max_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                rom_param['cons_proj_max_ma_counter']   = np.zeros(solver_param['num_state_var'])

                rom_param['cons_proj_avg_short_ma']     = MovingAverage(short_ma_window, solver_param['num_state_var'])
                rom_param['cons_proj_avg_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                rom_param['cons_proj_avg_ma_counter']   = np.zeros(solver_param['num_state_var'])

                rom_param['prim_interp_max_short_ma']   = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                rom_param['prim_interp_max_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                rom_param['prim_interp_max_ma_counter'] = np.zeros(solver_param['num_prim_var'])

                rom_param['prim_interp_avg_short_ma']   = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                rom_param['prim_interp_avg_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                rom_param['prim_interp_avg_ma_counter'] = np.zeros(solver_param['num_prim_var'])

                rom_param['prim_proj_max_short_ma']     = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                rom_param['prim_proj_max_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                rom_param['prim_proj_max_ma_counter']   = np.zeros(solver_param['num_prim_var'])

                rom_param['prim_proj_avg_short_ma']     = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                rom_param['prim_proj_avg_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                rom_param['prim_proj_avg_ma_counter']   = np.zeros(solver_param['num_prim_var'])

            else:
                # Files should already be open. If they are not, open in append mode.
                mode = "a"

                if "error_output_files" not in rom_param:
                    rom_param["error_output_flush_interval"] = solver_param.get("error_output_flush_interval", 100)
                    _open_error_output_files(rom_param, dir_results, mode)

                # Compute the error slope
                cons_interp_max_slope = cons_interp_max - rom_param['cons_interp_max_store']
                cons_interp_avg_slope = cons_interp_avg - rom_param['cons_interp_avg_store']
                cons_proj_max_slope = cons_proj_max - rom_param['cons_proj_max_store']
                cons_proj_avg_slope = cons_proj_avg - rom_param['cons_proj_avg_store']

                prim_interp_max_slope = prim_interp_max - rom_param['prim_interp_max_store']
                prim_interp_avg_slope = prim_interp_avg - rom_param['prim_interp_avg_store']
                prim_proj_max_slope = prim_proj_max - rom_param['prim_proj_max_store']
                prim_proj_avg_slope = prim_proj_avg - rom_param['prim_proj_avg_store']

                # Find the sign and add to the counter
                # rom_param['cons_interp_max_slope_counter'] += np.sign(cons_interp_max_slope)
                # rom_param['cons_interp_avg_slope_counter'] += np.sign(cons_interp_avg_slope)
                # rom_param['cons_proj_max_slope_counter'] += np.sign(cons_proj_max_slope)
                # rom_param['cons_proj_avg_slope_counter'] += np.sign(cons_proj_avg_slope)

                # rom_param['prim_interp_max_slope_counter'] += np.sign(prim_interp_max_slope)
                # rom_param['prim_interp_avg_slope_counter'] += np.sign(prim_interp_avg_slope)
                # rom_param['prim_proj_max_slope_counter'] += np.sign(prim_proj_max_slope)
                # rom_param['prim_proj_avg_slope_counter'] += np.sign(prim_proj_avg_slope)

                rom_param['cons_interp_max_slope_counter'] += np.sign(cons_interp_max_slope) * slope_weight(cons_interp_max, rom_param['cons_interp_max_store'], cons_interp_max_slope)
                rom_param['cons_interp_avg_slope_counter'] += np.sign(cons_interp_avg_slope) * slope_weight(cons_interp_avg, rom_param['cons_interp_avg_store'], cons_interp_avg_slope)
                rom_param['cons_proj_max_slope_counter']   += np.sign(cons_proj_max_slope)   * slope_weight(cons_proj_max,   rom_param['cons_proj_max_store'],   cons_proj_max_slope)
                rom_param['cons_proj_avg_slope_counter']   += np.sign(cons_proj_avg_slope)   * slope_weight(cons_proj_avg,   rom_param['cons_proj_avg_store'],   cons_proj_avg_slope)

                rom_param['prim_interp_max_slope_counter'] += np.sign(prim_interp_max_slope) * slope_weight(prim_interp_max, rom_param['prim_interp_max_store'], prim_interp_max_slope)
                rom_param['prim_interp_avg_slope_counter'] += np.sign(prim_interp_avg_slope) * slope_weight(prim_interp_avg, rom_param['prim_interp_avg_store'], prim_interp_avg_slope)
                rom_param['prim_proj_max_slope_counter']   += np.sign(prim_proj_max_slope)   * slope_weight(prim_proj_max,   rom_param['prim_proj_max_store'],   prim_proj_max_slope)
                rom_param['prim_proj_avg_slope_counter']   += np.sign(prim_proj_avg_slope)   * slope_weight(prim_proj_avg,   rom_param['prim_proj_avg_store'],   prim_proj_avg_slope)

                cons_interp_max_slope_ratio = np.where(cons_interp_max_slope > 0, cons_interp_max/rom_param['cons_interp_max_store'], np.where(cons_interp_max_slope < 0, rom_param['cons_interp_max_store']/cons_interp_max, 0))
                cons_interp_avg_slope_ratio = np.where(cons_interp_avg_slope > 0, cons_interp_avg/rom_param['cons_interp_avg_store'], np.where(cons_interp_avg_slope < 0, rom_param['cons_interp_avg_store']/cons_interp_avg, 0))
                cons_proj_max_slope_ratio   = np.where(cons_proj_max_slope   > 0, cons_proj_max/rom_param['cons_proj_max_store'],     np.where(cons_proj_max_slope   < 0, rom_param['cons_proj_max_store']/cons_proj_max,     0))
                cons_proj_avg_slope_ratio   = np.where(cons_proj_avg_slope   > 0, cons_proj_avg/rom_param['cons_proj_avg_store'],     np.where(cons_proj_avg_slope   < 0, rom_param['cons_proj_avg_store']/cons_proj_avg,     0))

                prim_interp_max_slope_ratio = np.where(prim_interp_max_slope > 0, prim_interp_max/rom_param['prim_interp_max_store'], np.where(prim_interp_max_slope < 0, rom_param['prim_interp_max_store']/prim_interp_max, 0))
                prim_interp_avg_slope_ratio = np.where(prim_interp_avg_slope > 0, prim_interp_avg/rom_param['prim_interp_avg_store'], np.where(prim_interp_avg_slope < 0, rom_param['prim_interp_avg_store']/prim_interp_avg, 0))
                prim_proj_max_slope_ratio   = np.where(prim_proj_max_slope   > 0, prim_proj_max/rom_param['prim_proj_max_store'],     np.where(prim_proj_max_slope   < 0, rom_param['prim_proj_max_store']/prim_proj_max,     0))
                prim_proj_avg_slope_ratio   = np.where(prim_proj_avg_slope   > 0, prim_proj_avg/rom_param['prim_proj_avg_store'],     np.where(prim_proj_avg_slope   < 0, rom_param['prim_proj_avg_store']/prim_proj_avg,     0))

            # Store the current error QoIs
            rom_param['cons_interp_max_store'] = cons_interp_max
            rom_param['cons_interp_avg_store'] = cons_interp_avg
            rom_param['cons_proj_max_store'] = cons_proj_max
            rom_param['cons_proj_avg_store'] = cons_proj_avg

            rom_param['prim_interp_max_store'] = prim_interp_max
            rom_param['prim_interp_avg_store'] = prim_interp_avg
            rom_param['prim_proj_max_store'] = prim_proj_max
            rom_param['prim_proj_avg_store'] = prim_proj_avg

            ## Update the moving averages
            rom_param['cons_interp_max_short_ma'].update(cons_interp_max)
            rom_param['cons_interp_max_long_ma'].update(cons_interp_max)

            rom_param['cons_interp_avg_short_ma'].update(cons_interp_avg)
            rom_param['cons_interp_avg_long_ma'].update(cons_interp_avg)

            rom_param['cons_proj_max_short_ma'].update(cons_proj_max)
            rom_param['cons_proj_max_long_ma'].update(cons_proj_max)

            rom_param['cons_proj_avg_short_ma'].update(cons_proj_avg)
            rom_param['cons_proj_avg_long_ma'].update(cons_proj_avg)

            rom_param['prim_interp_max_short_ma'].update(prim_interp_max)
            rom_param['prim_interp_max_long_ma'].update(prim_interp_max)

            rom_param['prim_interp_avg_short_ma'].update(prim_interp_avg)
            rom_param['prim_interp_avg_long_ma'].update(prim_interp_avg)

            rom_param['prim_proj_max_short_ma'].update(prim_proj_max)
            rom_param['prim_proj_max_long_ma'].update(prim_proj_max)

            rom_param['prim_proj_avg_short_ma'].update(prim_proj_avg)
            rom_param['prim_proj_avg_long_ma'].update(prim_proj_avg)

            ## Update the moving average counter
            rom_param['cons_interp_max_ma_counter'] += np.sign(rom_param['cons_interp_max_short_ma'].avg - rom_param['cons_interp_max_long_ma'].avg)
            rom_param['cons_interp_avg_ma_counter'] += np.sign(rom_param['cons_interp_avg_short_ma'].avg - rom_param['cons_interp_avg_long_ma'].avg)
            rom_param['cons_proj_max_ma_counter']   += np.sign(rom_param['cons_proj_max_short_ma'].avg   - rom_param['cons_proj_max_long_ma'].avg)
            rom_param['cons_proj_avg_ma_counter']   += np.sign(rom_param['cons_proj_avg_short_ma'].avg   - rom_param['cons_proj_avg_long_ma'].avg)

            rom_param['prim_interp_max_ma_counter'] += np.sign(rom_param['prim_interp_max_short_ma'].avg - rom_param['prim_interp_max_long_ma'].avg)
            rom_param['prim_interp_avg_ma_counter'] += np.sign(rom_param['prim_interp_avg_short_ma'].avg - rom_param['prim_interp_avg_long_ma'].avg)
            rom_param['prim_proj_max_ma_counter']   += np.sign(rom_param['prim_proj_max_short_ma'].avg   - rom_param['prim_proj_max_long_ma'].avg)
            rom_param['prim_proj_avg_ma_counter']   += np.sign(rom_param['prim_proj_avg_short_ma'].avg   - rom_param['prim_proj_avg_long_ma'].avg)

            rom_param['cons_interp_max_ma_counter_sum'] = np.sum(rom_param['cons_interp_max_ma_counter'])
            rom_param['cons_interp_avg_ma_counter_sum'] = np.sum(rom_param['cons_interp_avg_ma_counter'])
            rom_param['cons_proj_max_ma_counter_sum']   = np.sum(rom_param['cons_proj_max_ma_counter'])
            rom_param['cons_proj_avg_ma_counter_sum']   = np.sum(rom_param['cons_proj_avg_ma_counter'])

            rom_param['prim_interp_max_ma_counter_sum'] = np.sum(rom_param['prim_interp_max_ma_counter'])
            rom_param['prim_interp_avg_ma_counter_sum'] = np.sum(rom_param['prim_interp_avg_ma_counter'])
            rom_param['prim_proj_max_ma_counter_sum']   = np.sum(rom_param['prim_proj_max_ma_counter'])
            rom_param['prim_proj_avg_ma_counter_sum']   = np.sum(rom_param['prim_proj_avg_ma_counter'])

            # # Save the current errors for gradient calculation
            # rom_param['Q_cons_interp_error_save'] = Q_cons_interp_error_reshape
            # rom_param['Q_prim_interp_error_save'] = Q_prim_interp_error_reshape
            # rom_param['Q_cons_proj_error_save'] = Q_cons_proj_error_reshape
            # rom_param['Q_prim_proj_error_save'] = Q_prim_proj_error_reshape

            # Write the full error vectors
            # _write_error_output_line(
            #     rom_param,
            #     "full_data/cons_interp_error",
            #     _format_array_line(iter, Q_cons_interp_error)
            # )

            # _write_error_output_line(
            #     rom_param,
            #     "full_data/prim_interp_error",
            #     _format_array_line(iter, Q_prim_interp_error)
            # )

            # _write_error_output_line(
            #     rom_param,
            #     "full_data/cons_proj_error",
            #     _format_array_line(iter, Q_cons_proj_error)
            # )

            # _write_error_output_line(
            #     rom_param,
            #     "full_data/prim_proj_error",
            #     _format_array_line(iter, Q_prim_proj_error)
            # )


            # Write error-output arrays using the centralized output registry
            qoi_data = {
                "prim_interp_max": prim_interp_max,
                "prim_interp_avg": prim_interp_avg,
                "prim_proj_max":   prim_proj_max,
                "prim_proj_avg":   prim_proj_avg,

                "cons_interp_max": cons_interp_max,
                "cons_interp_avg": cons_interp_avg,
                "cons_proj_max":   cons_proj_max,
                "cons_proj_avg":   cons_proj_avg,
            }

            slope_counter_data = {
                "cons_interp_max_slope_counter": rom_param["cons_interp_max_slope_counter"],
                "cons_interp_avg_slope_counter": rom_param["cons_interp_avg_slope_counter"],
                "cons_proj_max_slope_counter":   rom_param["cons_proj_max_slope_counter"],
                "cons_proj_avg_slope_counter":   rom_param["cons_proj_avg_slope_counter"],

                "prim_interp_max_slope_counter": rom_param["prim_interp_max_slope_counter"],
                "prim_interp_avg_slope_counter": rom_param["prim_interp_avg_slope_counter"],
                "prim_proj_max_slope_counter":   rom_param["prim_proj_max_slope_counter"],
                "prim_proj_avg_slope_counter":   rom_param["prim_proj_avg_slope_counter"],
            }

            slope_ratio_data = {
                "cons_interp_max_slope_ratio": cons_interp_max_slope_ratio,
                "cons_interp_avg_slope_ratio": cons_interp_avg_slope_ratio,
                "cons_proj_max_slope_ratio":   cons_proj_max_slope_ratio,
                "cons_proj_avg_slope_ratio":   cons_proj_avg_slope_ratio,

                "prim_interp_max_slope_ratio": prim_interp_max_slope_ratio,
                "prim_interp_avg_slope_ratio": prim_interp_avg_slope_ratio,
                "prim_proj_max_slope_ratio":   prim_proj_max_slope_ratio,
                "prim_proj_avg_slope_ratio":   prim_proj_avg_slope_ratio,
            }

            output_array_map = {}

            output_array_map.update(qoi_data)

            for name in qoi_data:
                output_array_map[f"{name}_short_ma"]   = rom_param[f"{name}_short_ma"].avg
                output_array_map[f"{name}_long_ma"]    = rom_param[f"{name}_long_ma"].avg
                output_array_map[f"{name}_ma_counter"] = rom_param[f"{name}_ma_counter"]
                output_array_map[f"{name}_ma_counter_sum"] = rom_param[f"{name}_ma_counter_sum"]

            output_array_map.update(slope_counter_data)
            output_array_map.update(slope_ratio_data)

            _write_error_output_array_map(
                rom_param,
                iter,
                output_array_map,
            )

            # Write sampling frequency separately because it is scalar data
            _write_error_output_line(
                rom_param,
                "sampling_freq",
                str(iter) + "," + str(solver_param["unsampled_update_freq"]) + "\n"
            )

            # Periodically flush the open file handles
            _flush_error_output_files(rom_param)

            # Check if the slope counter thresholds are exceeded
            # if solver_param['parameter_adapt']:
            #     if np.any((rom_param['prim_interp_max_slope_counter'] >= 100) | (rom_param['prim_interp_max_slope_counter'] <= -100)):

            #         if np.any(rom_param['prim_interp_max_slope_counter'] >= 100):
            #             if solver_param['unsampled_update_freq'] == 2:
            #                 print('Update frequency is already at max')
            #             else:
            #                 solver_param['unsampled_update_freq'] -= 1
            #         else:
            #             solver_param['unsampled_update_freq'] += 1

            #         print('Updated the sampling frequency to', solver_param['unsampled_update_freq'])

            #         # Update the sampling iterations
            #         past_samples = solver_param['resample_iter_list'][solver_param['resample_iter_list'] <= iter]

            #         if np.any(rom_param['prim_interp_max_slope_counter'] >= 100):
            #             # Set a sampling update at the next iteration
            #             future_samples = np.arange(
            #                 iter + 1,
            #                 solver_param['num_step'],
            #                 solver_param['unsampled_update_freq'],
            #                 dtype=int
            #             )
            #         else:
            #             future_samples = np.arange(
            #                 iter + solver_param['unsampled_update_freq'],
            #                 solver_param['num_step'],
            #                 solver_param['unsampled_update_freq'],
            #                 dtype=int
            #             )

            #         solver_param['resample_iter_list'] = np.concatenate((past_samples, future_samples))

            #         # Reset the slope counter
            #         rom_param['cons_interp_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         rom_param['cons_interp_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         rom_param['cons_proj_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         rom_param['cons_proj_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])

            #         rom_param['prim_interp_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         rom_param['prim_interp_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         rom_param['prim_proj_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         rom_param['prim_proj_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])

    if ( solver_param['error_check'] and not ( np.any(solver_param['iter'] == solver_param['resample_iter_list']) ) ):
        Q_cons_reshape = np.sum(np.reshape(state['Q_cons'],[solver_param['num_state_var'],solver_param['cell_number']+4])[:,2:-2],axis=1)
        error_cons = Q_cons_reshape - (np.sum(np.reshape(state['Q_cons_pre'],[solver_param['num_state_var'],solver_param['cell_number']+4])[:,2:-2],axis=1) + state['Q_cons_inj'])
        error_cons_perct = 100.0 * error_cons/Q_cons_reshape

        # This is outside the periodically skipped interpolation/projection
        # error block, so both values are exported at every timestep.
        dir_results = os.path.join(solver_param["dir_results"], "error")

        if "conservation_error_output_files" not in state:
            state["error_output_flush_interval"] = solver_param.get(
                "error_output_flush_interval",
                100,
            )
            output_mode = solver_param.get(
                "conservation_error_output_mode",
                "w",
            )
            _open_conservation_error_output_files(
                state,
                dir_results,
                output_mode,
            )

        _write_conservation_error_output_line(
            state,
            "error_cons",
            _format_array_line(iter, error_cons),
        )
        _write_conservation_error_output_line(
            state,
            "error_cons_perct",
            _format_array_line(iter, error_cons_perct),
        )
        _flush_conservation_error_output_files(state)

    return solver_param, state, rom_param