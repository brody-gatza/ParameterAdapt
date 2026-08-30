import os
import numpy as np


from compflowlab.boundary_condition import bc_func
from compflowlab.utils import reshape_func


def prepare_to_store(solver_param,state,rom_param=None):

    # prepare data to save
    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]
    state['res_save']          = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['d_flux_dx'])[:,2:-2]
    
    if solver_param['gas_model'] == 'Air':

        state['prim_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]


    return state

def results_recorder(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results, f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results, f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results, f"{save_title}_res.npy") , state['res_save'])

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    if solver_param['error_check']:
        state['Q_cons_pre'] = state['Q_cons'].copy()
        state['Q_cons_inj'] = np.zeros(solver_param['num_state_var'])

    # compute residual
    state = physics.residual_calculator(solver_param,rom_param,state)

    # time integrate
    state = time_integration.advance_time(solver_param,rom_param,state,physics)

    if solver_param['injection']:

        state = physics.injection_correction(solver_param,state)

    # update prim state
    state = physics.cons2prim_converter(solver_param,state)

    # update the ghost cells
    state = bc_func.update_ghost_cell(solver_param,state)

    # update prim state
    state = physics.prim2cons_converter(solver_param,state)

    # prepare results to save
    state = prepare_to_store(solver_param,state,rom_param)

    # save the data 
    if solver_param['iter'] % solver_param['save_interval'] == 0:

        results_recorder(solver_param,state,rom_param)


    if solver_param['error_check']:
        Q_cons_reshape = np.sum(np.reshape(state['Q_cons'],[solver_param['num_state_var'],solver_param['cell_number']+4])[:,2:-2],axis=1)
        error_cons = Q_cons_reshape - (np.sum(np.reshape(state['Q_cons_pre'],[solver_param['num_state_var'],solver_param['cell_number']+4])[:,2:-2],axis=1) + state['Q_cons_inj'])
        error_cons_perct = 100.0 * error_cons/Q_cons_reshape

        dir_error = os.path.join(solver_param['dir_results'], 'error')
        os.makedirs(dir_error, exist_ok=True)
        output_mode = 'w' if solver_param['iter'] == 0 else 'a'

        with open(os.path.join(dir_error, 'error_cons.txt'), output_mode) as file:
            file.write(str(solver_param['iter']) + ',' + ','.join(f"{value:.17e}" for value in np.atleast_1d(error_cons)) + '\n')

        with open(os.path.join(dir_error, 'error_cons_perct.txt'), output_mode) as file:
            file.write(str(solver_param['iter']) + ',' + ','.join(f"{value:.17e}" for value in np.atleast_1d(error_cons_perct)) + '\n')

    return solver_param, state, rom_param