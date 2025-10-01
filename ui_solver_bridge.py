
import solver_functions
import time_integrator_functions
import visualization_functions
import rom_functions
import numpy as np
import matplotlib.pyplot as plt
import time
import cantera as ct
import os
import shutil


def driver(solver_param):

    # breakpoint()
    
    # initialize main states 
    state = solver_functions.initialize_state(solver_param)

    if solver_param['injection']:

        state = solver_functions.injection_init(solver_param,state)

    # get the initial condition

    state = solver_functions.ic_generator(solver_param,state)

    # convert prim to cons
    state = solver_functions.prim2cons_converter(solver_param, state)

    # create folders for storing data
    dir_results = os.path.join(solver_param['working_dir'], f"{solver_param['solver_mode']}_results")
    solver_param['dir_results'] = dir_results

    # Check if the directory exists
    if os.path.exists(dir_results):

        # Remove the entire directory
        shutil.rmtree(dir_results)

    # Create a new directory
    os.makedirs(dir_results)

    # create rom related folders if we are running rom
    if solver_param['solver_mode'] != 'FOM':

        os.makedirs( os.path.join(dir_results, 'cons_prim')     )
        os.makedirs( os.path.join(dir_results, 'res')           )
        os.makedirs( os.path.join(dir_results, 'basis')         )
        os.makedirs( os.path.join(dir_results, 'samples_user')  )
        os.makedirs( os.path.join(dir_results, 'samples_solver'))
        os.makedirs( os.path.join(dir_results, 'q_r')           )
        os.makedirs( os.path.join(dir_results, 'q_ref')         )
        os.makedirs( os.path.join(dir_results, 'norm')          )
        os.makedirs( os.path.join(dir_results, 'denorm')        )

    # initialize rom if necessary
    if solver_param['solver_mode'] == 'FOM' or solver_param['solver_mode'] == 'Adaptive ROM' or solver_param['solver_mode'] == 'Hybrid ROM':

        S_indx_user               = np.arange(0,solver_param['cell_number'])
        S_indx_solver             = rom_functions.user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])
        pcc                       = 0
        solver_param['hyper']     = False

        rom_param = {}
        
        rom_param['S_indx_user']      = S_indx_user
        rom_param['S_indx_solver']    = S_indx_solver
        rom_param['hyper_precompute'] = pcc


    elif solver_param['solver_mode'] == 'ROM':

        rom_param = rom_functions.precomputer(solver_param,state)

        if solver_param['hyper'] == True:

            rom_param = rom_functions.sample_point_finder(solver_param,rom_param)

    if solver_param['visual'] == True:

        # create plot
        visual_param = visualization_functions.visual_var_collector(solver_param)

        plt.close()
        fig , axs = plt.subplots(2,2)
        fig.set_size_inches(15,6)
        
        visual_param        = visualization_functions.initial_plot(axs,solver_param,visual_param)

    # begin simulation
    start_time = time.time()
    state['time'] = 0

    solver_param['arom_restart']     = False
    # solver_param['basis_window_size']= solver_param['init_training_win']

    iter_list = np.arange(solver_param['num_step'])

    iter = 0

    while iter < iter_list[-1]:

        solver_param['iter'] = iter

        if solver_param['arom_restart']:

            try:

                if iter % solver_param['save_interval'] == 0:

                    checkpoint = (
                                solver_param.copy(), rom_param.copy(), state.copy(),
                                fig, axs, visual_param.copy()
                            )
                    
                    checkpoint_iter = iter

                solver_param,rom_param,state,fig,axs,visual_param = solver_functions.advance_one_time_step(solver_param,rom_param,state,fig,axs,visual_param)

            except:

                print('simulation failed, rolling back and restarting!')

                solver_param, rom_param, state, fig, axs, visual_param = checkpoint

                iter = checkpoint_iter
                solver_param['init_training_win'] = iter + 10

                state['Q_cons']           = checkpoint[2]['Q_cons']
                state['Q_prim']           = checkpoint[2]['Q_prim']
                solver_param['hyper']     = False


        else:

            solver_param,rom_param,state,fig,axs,visual_param = solver_functions.advance_one_time_step(solver_param,rom_param,state,fig,axs,visual_param)

        iter = iter + 1
    # breakpoint()

    # np.save(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\cumsum.npy'    ,rom_param['cum_sum'])
    # np.save(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\moving_avg.npy',rom_param['moving_avg'])
    # np.save(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\sub_angle.npy' ,rom_param['subspace_angle'])

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    print('Simulation successfully completed!')

    return state



         

 
        






        







        


        
