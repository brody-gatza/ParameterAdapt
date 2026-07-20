import os
import numpy as np


from compflowlab.utils import reshape_func
from compflowlab.boundary_condition import bc_func
from compflowlab.rom.basis_func import adapt_basis
from compflowlab.rom.sampling_func import hyper_precompute

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
    dir_results = os.path.join(solver_param['dir_results'] + "/error/")
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

            # if solver_param['gas_model'] == 'Air':
            Q_prim_interp_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_interp_error)[:,2:-2]
            Q_prim_proj_error_reshape = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_proj_error)[:,2:-2]
            Q_prim_FOM_max = np.max(reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim_FOM)[:,2:-2], axis=1)

            # Is heat release not stored in the state vector?    
            # else:
            #     state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
            #     state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

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
            dir_results = os.path.join(solver_param['dir_results'] + "/error/")
            if iter == int(solver_param['FOM2ROM_trans_iter']) + 1:
                mode = "w"

                gradient_files = [
                    "grad_prim_interp_max.txt",
                    "grad_prim_interp_min.txt",
                    "grad_prim_interp_avg.txt",
                    "grad_prim_proj_max.txt",
                    "grad_prim_proj_min.txt",
                    "grad_prim_proj_avg.txt",
                    "grad_cons_interp_max.txt",
                    "grad_cons_interp_min.txt",
                    "grad_cons_interp_avg.txt",
                    "grad_cons_proj_max.txt",
                    "grad_cons_proj_min.txt",
                    "grad_cons_proj_avg.txt",
                ]

                for filename in gradient_files:
                    open(dir_results + filename, "w").close()

            else:
                mode = "a"

                # # Calculate the error gradient
                # grad_cons_interp = np.abs(Q_cons_interp_error_reshape - state['Q_cons_interp_error_save'])
                # grad_prim_interp = np.abs(Q_prim_interp_error_reshape - state['Q_prim_interp_error_save'])
                # grad_cons_proj = np.abs(Q_cons_proj_error_reshape - state['Q_cons_proj_error_save'])
                # grad_prim_proj = np.abs(Q_prim_proj_error_reshape - state['Q_prim_proj_error_save'])
            
                # # Calculate the gradient QoIs per variable
                # grad_cons_interp_max = np.max(grad_cons_interp, axis=1)
                # grad_cons_interp_min = np.min(grad_cons_interp, axis=1)
                # grad_cons_interp_avg = np.mean(grad_cons_interp, axis=1)
                # grad_cons_proj_max = np.max(grad_cons_proj, axis=1)
                # grad_cons_proj_min = np.min(grad_cons_proj, axis=1)
                # grad_cons_proj_avg = np.mean(grad_cons_proj, axis=1)

                # grad_prim_interp_max = np.max(grad_prim_interp, axis=1)
                # grad_prim_interp_min = np.min(grad_prim_interp, axis=1)
                # grad_prim_interp_avg = np.mean(grad_prim_interp, axis=1)
                # grad_prim_proj_max = np.max(grad_prim_proj, axis=1)
                # grad_prim_proj_min = np.min(grad_prim_proj, axis=1)
                # grad_prim_proj_avg = np.mean(grad_prim_proj, axis=1)

            # # Save the current errors for gradient calculation
            # state['Q_cons_interp_error_save'] = Q_cons_interp_error_reshape
            # state['Q_prim_interp_error_save'] = Q_prim_interp_error_reshape
            # state['Q_cons_proj_error_save'] = Q_cons_proj_error_reshape
            # state['Q_prim_proj_error_save'] = Q_prim_proj_error_reshape

            # Write the full error vectors
            # with open(dir_results/full_data/ + "cons_interp_error.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in Q_cons_interp_error) + "\n")

            # with open(dir_results/full_data/ + "prim_interp_error.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in Q_prim_interp_error) + "\n")

            # with open(dir_results/full_data/ + "cons_proj_error.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in Q_cons_proj_error) + "\n")

            # with open(dir_results/full_data/ + "prim_proj_error.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in Q_prim_proj_error) + "\n")

            # Write the QoIs
            with open(dir_results + "prim_interp_max.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_interp_max) + "\n")
                
            # with open(dir_results + "prim_interp_min.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_interp_min) + "\n")

            with open(dir_results + "prim_interp_avg.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_interp_avg) + "\n")

            with open(dir_results + "prim_proj_max.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_proj_max) + "\n")
                
            # with open(dir_results + "prim_proj_min.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_proj_min) + "\n")

            with open(dir_results + "prim_proj_avg.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in prim_proj_avg) + "\n")

            with open(dir_results + "cons_interp_max.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_interp_max) + "\n")
                
            # with open(dir_results + "cons_interp_min.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_interp_min) + "\n")

            with open(dir_results + "cons_interp_avg.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_interp_avg) + "\n")

            with open(dir_results + "cons_proj_max.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_proj_max) + "\n")
                
            # with open(dir_results + "cons_proj_min.txt", mode) as file:
            #     file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_proj_min) + "\n")

            with open(dir_results + "cons_proj_avg.txt", mode) as file:
                file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in cons_proj_avg) + "\n")
            
            # if iter > int(solver_param['FOM2ROM_trans_iter']) + 1:
            #     with open(dir_results + "grad_prim_interp_max.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_interp_max) + "\n")
                    
            #     with open(dir_results + "grad_prim_interp_min.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_interp_min) + "\n")

            #     with open(dir_results + "grad_prim_interp_avg.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_interp_avg) + "\n")

            #     with open(dir_results + "grad_prim_proj_max.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_proj_max) + "\n")
                    
            #     with open(dir_results + "grad_prim_proj_min.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_proj_min) + "\n")

            #     with open(dir_results + "grad_prim_proj_avg.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_prim_proj_avg) + "\n")

            #     with open(dir_results + "grad_cons_interp_max.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_interp_max) + "\n")
                    
            #     with open(dir_results + "grad_cons_interp_min.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_interp_min) + "\n")

            #     with open(dir_results + "grad_cons_interp_avg.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_interp_avg) + "\n")

            #     with open(dir_results + "grad_cons_proj_max.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_proj_max) + "\n")
                    
            #     with open(dir_results + "grad_cons_proj_min.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_proj_min) + "\n")

            #     with open(dir_results + "grad_cons_proj_avg.txt", mode) as file:
            #         file.write(str(iter) + "," + ",".join(f"{x:.17e}" for x in grad_cons_proj_avg) + "\n")
        # Update Samples
        # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0) or (iter == int(solver_param['init_training_win'])+1):
        # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0):
        if np.any(solver_param['iter'] == solver_param['resample_iter_list']):

            rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

    return solver_param, state, rom_param