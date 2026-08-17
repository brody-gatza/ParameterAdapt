import os
import numpy as np


from compflowlab.utils import reshape_func
from compflowlab.boundary_condition import bc_func
from compflowlab.rom.basis_func import adapt_basis
from compflowlab.rom.sampling_func import hyper_precompute

from ..utils.classes import MovingAverage

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
    return str(iter) + "," + ",".join(f"{x:.17e}" for x in arr) + "\n"


def _open_error_output_files(state, dir_results, mode):
    """
    Opens error-output files once and stores file handles in state.

    mode should be:
        "w" on the first error-output iteration
        "a" if restarting/appending
    """

    file_names = [
        "prim_interp_max",
        "prim_interp_avg",
        "prim_proj_max",
        "prim_proj_avg",

        "cons_interp_max",
        "cons_interp_avg",
        "cons_proj_max",
        "cons_proj_avg",

        "cons_interp_max_slope_counter",
        "cons_interp_avg_slope_counter",
        "cons_proj_max_slope_counter",
        "cons_proj_avg_slope_counter",

        "prim_interp_max_slope_counter",
        "prim_interp_avg_slope_counter",
        "prim_proj_max_slope_counter",
        "prim_proj_avg_slope_counter",

        "cons_interp_max_slope_ratio",
        "cons_interp_avg_slope_ratio",
        "cons_proj_max_slope_ratio",
        "cons_proj_avg_slope_ratio",

        "prim_interp_max_slope_ratio",
        "prim_interp_avg_slope_ratio",
        "prim_proj_max_slope_ratio",
        "prim_proj_avg_slope_ratio",

        "cons_interp_max_short_ma",
        "cons_interp_avg_short_ma",
        "cons_proj_max_short_ma",
        "cons_proj_avg_short_ma",

        "prim_interp_max_short_ma",
        "prim_interp_avg_short_ma",
        "prim_proj_max_short_ma",
        "prim_proj_avg_short_ma",

        "cons_interp_max_long_ma",
        "cons_interp_avg_long_ma",
        "cons_proj_max_long_ma",
        "cons_proj_avg_long_ma",

        "prim_interp_max_long_ma",
        "prim_interp_avg_long_ma",
        "prim_proj_max_long_ma",
        "prim_proj_avg_long_ma",

        "cons_interp_max_ma_counter",
        "cons_interp_avg_ma_counter",
        "cons_proj_max_ma_counter",
        "cons_proj_avg_ma_counter",

        "prim_interp_max_ma_counter",
        "prim_interp_avg_ma_counter",
        "prim_proj_max_ma_counter",
        "prim_proj_avg_ma_counter",

        "sampling_freq",

        # "full_data/cons_interp_error",
        # "full_data/prim_interp_error",
        # "full_data/cons_proj_error",
        # "full_data/prim_proj_error",
    ]

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

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    #############################################
    # This function is taking one time step     #
    # using adaptive ROM algorithm. It will run #
    # FOM initially but then will turn into ROM #
    # with evolving basis and sample adaptation #
    #############################################

    iter = solver_param['iter']

    if iter <= int(solver_param['FOM2ROM_trans_iter']):

        if iter != int(solver_param['FOM2ROM_trans_iter']):

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

            # save solution
            results_recorder_FOM(solver_param,state,rom_param)

        elif iter == int(solver_param['FOM2ROM_trans_iter']):

            # take one more FOM step and prepare basis and samples based on that
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

            # adjust the training range
            solver_param['training_start_iter'] = int(iter-solver_param['init_training_win'])
            solver_param['training_end_iter'  ] = iter
            solver_param['training_step_iter' ] = 1
            solver_param['training_data_dir']   = os.path.join(solver_param['dir_results'], 'cons_prim')

            # build ROM
            rom_param = precomputer(solver_param)

            # create a full-state copy
            state['Q_bar']     = state['Q_cons']
            rom_param['Q_bar'] = state['Q_bar'] 

            # adjust sampling configuration
            solver_param['hyper'] = True

            # find initial samples
            rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

            # create list of resampling time steps
            if solver_param['multi_samp']:
                solver_param['resample_iter_list'] = np.unique(np.concatenate((
                                                     np.arange(
                                                         solver_param['FOM2ROM_trans_iter'],
                                                         solver_param['multi_samp_iter'],
                                                         solver_param['unsampled_update_freq'],
                                                         dtype=int
                                                     ),
                                                     np.arange(
                                                         solver_param['multi_samp_iter'],
                                                         solver_param['num_step'],
                                                         solver_param['unsampled_update_freq_2'],
                                                         dtype=int
                                                     ))))
            else:
                solver_param['resample_iter_list'] = np.arange(solver_param['FOM2ROM_trans_iter'],
                                                               solver_param['num_step'],
                                                               solver_param['unsampled_update_freq'],dtype=int)

            # save ROM related param
            results_recorder_trans(solver_param,state,rom_param)

    else:

        # read basic parameters
        q_ref                  = rom_param['q_ref']
        normalizor             = rom_param['norm']
        denormalizor           = rom_param['denorm']

        # Q tilda (ROM) before any update
        Q_tilda_old            = state['Q_cons']
        Q_tilda_old_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_old)

        # This routine evalutes the FOM based on the current time step to calculate ROM errors
        if solver_param['error_check']:

            # disable hyper-reduction so the FOM is fully evaluated
            Q_cons_save = state['Q_cons']
            Q_prim_save = state['Q_prim']
            hyper_save = solver_param['hyper']
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

            # reset solver parameters and store the FOM solution
            Q_cons_FOM = state['Q_cons']
            Q_prim_FOM = state['Q_prim']
            state['Q_cons'] = Q_cons_save
            state['Q_prim'] = Q_prim_save
            solver_param['hyper'] = hyper_save

        # it must be decided to whether to a large time step FOM to update whole domain or 
        # perform a normal time step ROM
        if solver_param['multi_samp']:
            if solver_param['iter'] >= solver_param['multi_samp_iter']:
                sampling_adapt_freq = solver_param['unsampled_update_freq_2']
            else:
                sampling_adapt_freq = solver_param['unsampled_update_freq']
        else:
            sampling_adapt_freq = solver_param['unsampled_update_freq']

        # condition for large time step FOM
        # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0):
        if np.any(solver_param['iter'] == solver_param['resample_iter_list']):

            # adjust the solver parameters for taking large time step FOM
            Q_bar_star_old        = state['Q_bar']
            solver_param['hyper'] = False
            solver_param['dt']    = sampling_adapt_freq * solver_param['dt']
            state['Q_cons']       = Q_bar_star_old

            # take one FOM step
            state = physics.residual_calculator(solver_param,rom_param,state)
            state = time_integration.advance_time(solver_param,rom_param,state,physics)

            if solver_param['injection']: 

                state = physics.injection_correction(solver_param,state)

            Q_bar_star_new = state['Q_cons']
            Q_bar_star_new_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_star_new)
            
            # update the large time step state 
            state['Q_bar'] = Q_bar_star_new

            # some sampling methods (ex. FGS) need this for sampling
            rom_param['Q_bar'] = state['Q_bar']

            # hold new solution
            Q_bar_star_new            = state['Q_cons']
            Q_bar_star_new_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_star_new)

            # reset solver parameters to samller time step setup (user defined setup)
            # Q_bar_new_sampling      = Q_bar_star_new_solver_int[rom_param['S_indx_solver']]
            Q_bar_new_solver_int    = Q_bar_star_new_solver_int
            solver_param['hyper']   = True
            solver_param['dt']      = solver_param['dt'] / sampling_adapt_freq

            # adapt basis with newly found sanpshot
            rom_param = adapt_basis(solver_param,rom_param,Q_bar_new_solver_int)
        
            # find corrected qr (projected with new basis)
            new_qr = np.transpose(rom_param['basis']) @ rom_param['F'][:,-1]

            # update states 
            corrected_cent_norm = rom_param['basis'] @ new_qr
            rom_param['F'][:,-1]   = corrected_cent_norm
            rom_param['Q_R'][:,-1] = new_qr

            # find new solution 
            Q_tilda_correct_solver_int = q_ref + (denormalizor * corrected_cent_norm)
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
            state = prepare_to_store_FOM(solver_param,state,rom_param)

            # save the data 
            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_ROM(solver_param,state,rom_param)

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
        if solver_param['error_check']:

            # Calculate the interpolation error
            # Crude, inclues FOM evaluations at sampled points but should be good enough
            Q_cons_interp_error = np.abs(Q_cons_FOM - state['Q_cons'])
            Q_prim_interp_error = np.abs(Q_prim_FOM - state['Q_prim'])

            # Calculate the projection error 
            Q_cons_FOM_int = normalizor * (reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_cons_FOM) - q_ref)
            Q_cons_FOM_int_proj = denormalizor * (rom_param['basis'] @ (rom_param['basis'].T @ Q_cons_FOM_int)) + q_ref
            Q_cons_FOM_proj = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_cons_FOM_int_proj)
            Q_cons_proj_error = np.abs(Q_cons_FOM - Q_cons_FOM_proj)

            # Save the ROM states
            Q_cons_save = state['Q_cons']
            Q_prim_save = state['Q_prim']

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

            # Reset the the ROM states
            state['Q_cons'] = Q_cons_save
            state['Q_prim'] = Q_prim_save

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

            # Compute moving averages


            # Write the error values
            dir_results = os.path.join(solver_param["dir_results"], "error")
            if iter == int(solver_param['FOM2ROM_trans_iter']) + 1:
                # Creates new files or clears existing files
                mode = "w"

                state["error_output_flush_interval"] = solver_param.get("error_output_flush_interval", 100)

                if "error_output_files" not in state:
                    _open_error_output_files(state, dir_results, mode)

                state['cons_interp_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
                state['cons_interp_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])
                state['cons_proj_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
                state['cons_proj_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])

                state['prim_interp_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                state['prim_interp_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                state['prim_proj_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
                state['prim_proj_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])

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

                state['cons_interp_max_short_ma']   = MovingAverage(short_ma_window, solver_param['num_state_var'])
                state['cons_interp_max_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                state['cons_interp_max_ma_counter'] = np.zeros(solver_param['num_state_var'])

                state['cons_interp_avg_short_ma']   = MovingAverage(short_ma_window, solver_param['num_state_var'])
                state['cons_interp_avg_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                state['cons_interp_avg_ma_counter'] = np.zeros(solver_param['num_state_var'])

                state['cons_proj_max_short_ma']     = MovingAverage(short_ma_window, solver_param['num_state_var'])
                state['cons_proj_max_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                state['cons_proj_max_ma_counter']   = np.zeros(solver_param['num_state_var'])

                state['cons_proj_avg_short_ma']     = MovingAverage(short_ma_window, solver_param['num_state_var'])
                state['cons_proj_avg_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_state_var'])
                state['cons_proj_avg_ma_counter']   = np.zeros(solver_param['num_state_var'])

                state['prim_interp_max_short_ma']   = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                state['prim_interp_max_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                state['prim_interp_max_ma_counter'] = np.zeros(solver_param['num_prim_var'])

                state['prim_interp_avg_short_ma']   = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                state['prim_interp_avg_long_ma']    = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                state['prim_interp_avg_ma_counter'] = np.zeros(solver_param['num_prim_var'])

                state['prim_proj_max_short_ma']     = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                state['prim_proj_max_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                state['prim_proj_max_ma_counter']   = np.zeros(solver_param['num_prim_var'])

                state['prim_proj_avg_short_ma']     = MovingAverage(short_ma_window, solver_param['num_prim_var'])
                state['prim_proj_avg_long_ma']      = MovingAverage(long_ma_window,  solver_param['num_prim_var'])
                state['prim_proj_avg_ma_counter']   = np.zeros(solver_param['num_prim_var'])

            else:
                # Files should already be open. If they are not, open in append mode.
                mode = "a"

                if "error_output_files" not in state:
                    state["error_output_flush_interval"] = solver_param.get("error_output_flush_interval", 100)
                    _open_error_output_files(state, dir_results, mode)

                # Compute the error slope
                cons_interp_max_slope = cons_interp_max - state['cons_interp_max_store']
                cons_interp_avg_slope = cons_interp_avg - state['cons_interp_avg_store']
                cons_proj_max_slope = cons_proj_max - state['cons_proj_max_store']
                cons_proj_avg_slope = cons_proj_avg - state['cons_proj_avg_store']

                prim_interp_max_slope = prim_interp_max - state['prim_interp_max_store']
                prim_interp_avg_slope = prim_interp_avg - state['prim_interp_avg_store']
                prim_proj_max_slope = prim_proj_max - state['prim_proj_max_store']
                prim_proj_avg_slope = prim_proj_avg - state['prim_proj_avg_store']

                # Find the sign and add to the counter
                # state['cons_interp_max_slope_counter'] += np.sign(cons_interp_max_slope)
                # state['cons_interp_avg_slope_counter'] += np.sign(cons_interp_avg_slope)
                # state['cons_proj_max_slope_counter'] += np.sign(cons_proj_max_slope)
                # state['cons_proj_avg_slope_counter'] += np.sign(cons_proj_avg_slope)

                # state['prim_interp_max_slope_counter'] += np.sign(prim_interp_max_slope)
                # state['prim_interp_avg_slope_counter'] += np.sign(prim_interp_avg_slope)
                # state['prim_proj_max_slope_counter'] += np.sign(prim_proj_max_slope)
                # state['prim_proj_avg_slope_counter'] += np.sign(prim_proj_avg_slope)

                state['cons_interp_max_slope_counter'] += np.sign(cons_interp_max_slope) * slope_weight(cons_interp_max, state['cons_interp_max_store'], cons_interp_max_slope)
                state['cons_interp_avg_slope_counter'] += np.sign(cons_interp_avg_slope) * slope_weight(cons_interp_avg, state['cons_interp_avg_store'], cons_interp_avg_slope)
                state['cons_proj_max_slope_counter']   += np.sign(cons_proj_max_slope)   * slope_weight(cons_proj_max,   state['cons_proj_max_store'],   cons_proj_max_slope)
                state['cons_proj_avg_slope_counter']   += np.sign(cons_proj_avg_slope)   * slope_weight(cons_proj_avg,   state['cons_proj_avg_store'],   cons_proj_avg_slope)

                state['prim_interp_max_slope_counter'] += np.sign(prim_interp_max_slope) * slope_weight(prim_interp_max, state['prim_interp_max_store'], prim_interp_max_slope)
                state['prim_interp_avg_slope_counter'] += np.sign(prim_interp_avg_slope) * slope_weight(prim_interp_avg, state['prim_interp_avg_store'], prim_interp_avg_slope)
                state['prim_proj_max_slope_counter']   += np.sign(prim_proj_max_slope)   * slope_weight(prim_proj_max,   state['prim_proj_max_store'],   prim_proj_max_slope)
                state['prim_proj_avg_slope_counter']   += np.sign(prim_proj_avg_slope)   * slope_weight(prim_proj_avg,   state['prim_proj_avg_store'],   prim_proj_avg_slope)

                cons_interp_max_slope_ratio = np.where(cons_interp_max_slope > 0, cons_interp_max/state['cons_interp_max_store'], np.where(cons_interp_max_slope < 0, state['cons_interp_max_store']/cons_interp_max, 0))
                cons_interp_avg_slope_ratio = np.where(cons_interp_avg_slope > 0, cons_interp_avg/state['cons_interp_avg_store'], np.where(cons_interp_avg_slope < 0, state['cons_interp_avg_store']/cons_interp_avg, 0))
                cons_proj_max_slope_ratio   = np.where(cons_proj_max_slope   > 0, cons_proj_max/state['cons_proj_max_store'],     np.where(cons_proj_max_slope   < 0, state['cons_proj_max_store']/cons_proj_max,     0))
                cons_proj_avg_slope_ratio   = np.where(cons_proj_avg_slope   > 0, cons_proj_avg/state['cons_proj_avg_store'],     np.where(cons_proj_avg_slope   < 0, state['cons_proj_avg_store']/cons_proj_avg,     0))

                prim_interp_max_slope_ratio = np.where(prim_interp_max_slope > 0, prim_interp_max/state['prim_interp_max_store'], np.where(prim_interp_max_slope < 0, state['prim_interp_max_store']/prim_interp_max, 0))
                prim_interp_avg_slope_ratio = np.where(prim_interp_avg_slope > 0, prim_interp_avg/state['prim_interp_avg_store'], np.where(prim_interp_avg_slope < 0, state['prim_interp_avg_store']/prim_interp_avg, 0))
                prim_proj_max_slope_ratio   = np.where(prim_proj_max_slope   > 0, prim_proj_max/state['prim_proj_max_store'],     np.where(prim_proj_max_slope   < 0, state['prim_proj_max_store']/prim_proj_max,     0))
                prim_proj_avg_slope_ratio   = np.where(prim_proj_avg_slope   > 0, prim_proj_avg/state['prim_proj_avg_store'],     np.where(prim_proj_avg_slope   < 0, state['prim_proj_avg_store']/prim_proj_avg,     0))

            # Store the current error QoIs
            state['cons_interp_max_store'] = cons_interp_max
            state['cons_interp_avg_store'] = cons_interp_avg
            state['cons_proj_max_store'] = cons_proj_max
            state['cons_proj_avg_store'] = cons_proj_avg

            state['prim_interp_max_store'] = prim_interp_max
            state['prim_interp_avg_store'] = prim_interp_avg
            state['prim_proj_max_store'] = prim_proj_max
            state['prim_proj_avg_store'] = prim_proj_avg

            ## Update the moving averages
            state['cons_interp_max_short_ma'].update(cons_interp_max)
            state['cons_interp_max_long_ma'].update(cons_interp_max)

            state['cons_interp_avg_short_ma'].update(cons_interp_avg)
            state['cons_interp_avg_long_ma'].update(cons_interp_avg)

            state['cons_proj_max_short_ma'].update(cons_proj_max)
            state['cons_proj_max_long_ma'].update(cons_proj_max)

            state['cons_proj_avg_short_ma'].update(cons_proj_avg)
            state['cons_proj_avg_long_ma'].update(cons_proj_avg)

            state['prim_interp_max_short_ma'].update(prim_interp_max)
            state['prim_interp_max_long_ma'].update(prim_interp_max)

            state['prim_interp_avg_short_ma'].update(prim_interp_avg)
            state['prim_interp_avg_long_ma'].update(prim_interp_avg)

            state['prim_proj_max_short_ma'].update(prim_proj_max)
            state['prim_proj_max_long_ma'].update(prim_proj_max)

            state['prim_proj_avg_short_ma'].update(prim_proj_avg)
            state['prim_proj_avg_long_ma'].update(prim_proj_avg)

            ## Update the moving average counter
            state['cons_interp_max_ma_counter'] += np.sign(state['cons_interp_max_short_ma'].avg - state['cons_interp_max_long_ma'].avg)
            state['cons_interp_avg_ma_counter'] += np.sign(state['cons_interp_avg_short_ma'].avg - state['cons_interp_avg_long_ma'].avg)
            state['cons_proj_max_ma_counter']   += np.sign(state['cons_proj_max_short_ma'].avg   - state['cons_proj_max_long_ma'].avg)
            state['cons_proj_avg_ma_counter']   += np.sign(state['cons_proj_avg_short_ma'].avg   - state['cons_proj_avg_long_ma'].avg)

            state['prim_interp_max_ma_counter'] += np.sign(state['prim_interp_max_short_ma'].avg - state['prim_interp_max_long_ma'].avg)
            state['prim_interp_avg_ma_counter'] += np.sign(state['prim_interp_avg_short_ma'].avg - state['prim_interp_avg_long_ma'].avg)
            state['prim_proj_max_ma_counter']   += np.sign(state['prim_proj_max_short_ma'].avg   - state['prim_proj_max_long_ma'].avg)
            state['prim_proj_avg_ma_counter']   += np.sign(state['prim_proj_avg_short_ma'].avg   - state['prim_proj_avg_long_ma'].avg)

            # # Save the current errors for gradient calculation
            # state['Q_cons_interp_error_save'] = Q_cons_interp_error_reshape
            # state['Q_prim_interp_error_save'] = Q_prim_interp_error_reshape
            # state['Q_cons_proj_error_save'] = Q_cons_proj_error_reshape
            # state['Q_prim_proj_error_save'] = Q_prim_proj_error_reshape

            # Write the full error vectors
            # _write_error_output_line(
            #     state,
            #     "full_data/cons_interp_error",
            #     _format_array_line(iter, Q_cons_interp_error)
            # )

            # _write_error_output_line(
            #     state,
            #     "full_data/prim_interp_error",
            #     _format_array_line(iter, Q_prim_interp_error)
            # )

            # _write_error_output_line(
            #     state,
            #     "full_data/cons_proj_error",
            #     _format_array_line(iter, Q_cons_proj_error)
            # )

            # _write_error_output_line(
            #     state,
            #     "full_data/prim_proj_error",
            #     _format_array_line(iter, Q_prim_proj_error)
            # )


            # Write the QoIs
            _write_error_output_line(state, "prim_interp_max", _format_array_line(iter, prim_interp_max))

            # _write_error_output_line(state, "prim_interp_min", _format_array_line(iter, prim_interp_min))

            _write_error_output_line(state, "prim_interp_avg", _format_array_line(iter, prim_interp_avg))

            _write_error_output_line(state, "prim_proj_max", _format_array_line(iter, prim_proj_max))

            # _write_error_output_line(state, "prim_proj_min", _format_array_line(iter, prim_proj_min))

            _write_error_output_line(state, "prim_proj_avg", _format_array_line(iter, prim_proj_avg))

            _write_error_output_line(state, "cons_interp_max", _format_array_line(iter, cons_interp_max))

            # _write_error_output_line(state, "cons_interp_min", _format_array_line(iter, cons_interp_min))

            _write_error_output_line(state, "cons_interp_avg", _format_array_line(iter, cons_interp_avg))

            _write_error_output_line(state, "cons_proj_max", _format_array_line(iter, cons_proj_max))

            # _write_error_output_line(state, "cons_proj_min", _format_array_line(iter, cons_proj_min))

            _write_error_output_line(state, "cons_proj_avg", _format_array_line(iter, cons_proj_avg))

            # Write the short moving averages
            _write_error_output_line(
                state,
                "cons_interp_max_short_ma",
                _format_array_line(iter, state["cons_interp_max_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_interp_avg_short_ma",
                _format_array_line(iter, state["cons_interp_avg_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_proj_max_short_ma",
                _format_array_line(iter, state["cons_proj_max_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_proj_avg_short_ma",
                _format_array_line(iter, state["cons_proj_avg_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_interp_max_short_ma",
                _format_array_line(iter, state["prim_interp_max_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_interp_avg_short_ma",
                _format_array_line(iter, state["prim_interp_avg_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_proj_max_short_ma",
                _format_array_line(iter, state["prim_proj_max_short_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_proj_avg_short_ma",
                _format_array_line(iter, state["prim_proj_avg_short_ma"].avg)
            )


            # Write the long moving averages
            _write_error_output_line(
                state,
                "cons_interp_max_long_ma",
                _format_array_line(iter, state["cons_interp_max_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_interp_avg_long_ma",
                _format_array_line(iter, state["cons_interp_avg_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_proj_max_long_ma",
                _format_array_line(iter, state["cons_proj_max_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "cons_proj_avg_long_ma",
                _format_array_line(iter, state["cons_proj_avg_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_interp_max_long_ma",
                _format_array_line(iter, state["prim_interp_max_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_interp_avg_long_ma",
                _format_array_line(iter, state["prim_interp_avg_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_proj_max_long_ma",
                _format_array_line(iter, state["prim_proj_max_long_ma"].avg)
            )

            _write_error_output_line(
                state,
                "prim_proj_avg_long_ma",
                _format_array_line(iter, state["prim_proj_avg_long_ma"].avg)
            )


            # Write the moving average counters
            _write_error_output_line(
                state,
                "cons_interp_max_ma_counter",
                _format_array_line(iter, state["cons_interp_max_ma_counter"])
            )

            _write_error_output_line(
                state,
                "cons_interp_avg_ma_counter",
                _format_array_line(iter, state["cons_interp_avg_ma_counter"])
            )

            _write_error_output_line(
                state,
                "cons_proj_max_ma_counter",
                _format_array_line(iter, state["cons_proj_max_ma_counter"])
            )

            _write_error_output_line(
                state,
                "cons_proj_avg_ma_counter",
                _format_array_line(iter, state["cons_proj_avg_ma_counter"])
            )

            _write_error_output_line(
                state,
                "prim_interp_max_ma_counter",
                _format_array_line(iter, state["prim_interp_max_ma_counter"])
            )

            _write_error_output_line(
                state,
                "prim_interp_avg_ma_counter",
                _format_array_line(iter, state["prim_interp_avg_ma_counter"])
            )

            _write_error_output_line(
                state,
                "prim_proj_max_ma_counter",
                _format_array_line(iter, state["prim_proj_max_ma_counter"])
            )

            _write_error_output_line(
                state,
                "prim_proj_avg_ma_counter",
                _format_array_line(iter, state["prim_proj_avg_ma_counter"])
            )

            # Write the slope counters
            _write_error_output_line(
                state,
                "cons_interp_max_slope_counter",
                _format_array_line(iter, state["cons_interp_max_slope_counter"])
            )

            _write_error_output_line(
                state,
                "cons_interp_avg_slope_counter",
                _format_array_line(iter, state["cons_interp_avg_slope_counter"])
            )

            _write_error_output_line(
                state,
                "cons_proj_max_slope_counter",
                _format_array_line(iter, state["cons_proj_max_slope_counter"])
            )

            _write_error_output_line(
                state,
                "cons_proj_avg_slope_counter",
                _format_array_line(iter, state["cons_proj_avg_slope_counter"])
            )

            _write_error_output_line(
                state,
                "prim_interp_max_slope_counter",
                _format_array_line(iter, state["prim_interp_max_slope_counter"])
            )

            _write_error_output_line(
                state,
                "prim_interp_avg_slope_counter",
                _format_array_line(iter, state["prim_interp_avg_slope_counter"])
            )

            _write_error_output_line(
                state,
                "prim_proj_max_slope_counter",
                _format_array_line(iter, state["prim_proj_max_slope_counter"])
            )

            _write_error_output_line(
                state,
                "prim_proj_avg_slope_counter",
                _format_array_line(iter, state["prim_proj_avg_slope_counter"])
            )


            # Write the slope ratios
            _write_error_output_line(
                state,
                "cons_interp_max_slope_ratio",
                _format_array_line(iter, cons_interp_max_slope_ratio)
            )

            _write_error_output_line(
                state,
                "cons_interp_avg_slope_ratio",
                _format_array_line(iter, cons_interp_avg_slope_ratio)
            )

            _write_error_output_line(
                state,
                "cons_proj_max_slope_ratio",
                _format_array_line(iter, cons_proj_max_slope_ratio)
            )

            _write_error_output_line(
                state,
                "cons_proj_avg_slope_ratio",
                _format_array_line(iter, cons_proj_avg_slope_ratio)
            )

            _write_error_output_line(
                state,
                "prim_interp_max_slope_ratio",
                _format_array_line(iter, prim_interp_max_slope_ratio)
            )

            _write_error_output_line(
                state,
                "prim_interp_avg_slope_ratio",
                _format_array_line(iter, prim_interp_avg_slope_ratio)
            )

            _write_error_output_line(
                state,
                "prim_proj_max_slope_ratio",
                _format_array_line(iter, prim_proj_max_slope_ratio)
            )

            _write_error_output_line(
                state,
                "prim_proj_avg_slope_ratio",
                _format_array_line(iter, prim_proj_avg_slope_ratio)
            )


            # Write sampling frequency
            _write_error_output_line(
                state,
                "sampling_freq",
                str(iter) + "," + str(solver_param["unsampled_update_freq"]) + "\n"
            )

            # Periodically flush the open file handles
            _flush_error_output_files(state)

            # Check if the slope counter thresholds are exceeded
            # if solver_param['parameter_adapt']:
            #     if np.any((state['prim_interp_max_slope_counter'] >= 100) | (state['prim_interp_max_slope_counter'] <= -100)):

            #         if np.any(state['prim_interp_max_slope_counter'] >= 100):
            #             if solver_param['unsampled_update_freq'] == 2:
            #                 print('Update frequency is already at max')
            #             else:
            #                 solver_param['unsampled_update_freq'] -= 1
            #         else:
            #             solver_param['unsampled_update_freq'] += 1

            #         print('Updated the sampling frequency to', solver_param['unsampled_update_freq'])

            #         # Update the sampling iterations
            #         past_samples = solver_param['resample_iter_list'][solver_param['resample_iter_list'] <= iter]

            #         if np.any(state['prim_interp_max_slope_counter'] >= 100):
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
            #         state['cons_interp_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         state['cons_interp_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         state['cons_proj_max_slope_counter'] = np.zeros(solver_param['num_state_var'])
            #         state['cons_proj_avg_slope_counter'] = np.zeros(solver_param['num_state_var'])

            #         state['prim_interp_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         state['prim_interp_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         state['prim_proj_max_slope_counter'] = np.zeros(solver_param['num_prim_var'])
            #         state['prim_proj_avg_slope_counter'] = np.zeros(solver_param['num_prim_var'])


        # Update Samples
        # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0) or (iter == int(solver_param['init_training_win'])+1):
        # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0):
        if np.any(solver_param['iter'] == solver_param['resample_iter_list']):

            rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

    return solver_param, state, rom_param