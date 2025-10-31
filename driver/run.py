
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import init_func
from visual import visual_func
from utils import reshape_func

def run(solver_param):

    # initialize main states 
    state = init_func.init_state(solver_param)

    if solver_param['injection']:

        state = init_func.init_injection(solver_param,state)

    # initialize the physics module
    physics   = init_func.init_physics(solver_param)

    # initialize time integration module
    time_integration = init_func.init_time_integration(solver_param)

    # get the initial condition
    state     = init_func.ic_generator(solver_param,state)

    # convert prim to cons (initial condition in cons vars)
    state     = physics.prim2cons_converter(solver_param, state)

    # initialize solver module
    solver , rom_param, state   =  init_func.init_solver(solver_param,state)

    # update prims (in ROM cases cons can be replaced so prim update is needed)
    state                       = physics.cons2prim_converter(solver_param, state)

    # create folders for storing data
    init_func.init_dir(solver_param)

    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]

    # initialize the visualization
    if solver_param['visual']:

        # create plot
        visual_param = visual_func.visual_var_collector(solver_param)

        plt.close()
        fig , axs = plt.subplots(2,2)
        fig.set_size_inches(15,6)
        
        visual_param = visual_func.initial_plot(axs,visual_param)

    # begin simulation
    start_time = time.time()
    state['time'] = 0

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

                solver_param, state, rom_param = solver.advance_one_time_step(solver_param,state,physics,time_integration,rom_param)
                
            except:

                print('simulation failed, rolling back and restarting!')

                solver_param, rom_param, state, fig, axs, visual_param = checkpoint

                iter = checkpoint_iter
                solver_param['FOM2ROM_trans_iter'] = int(iter + solver_param['init_training_win'])

                state['Q_cons']           = checkpoint[2]['Q_cons']
                state['Q_prim']           = checkpoint[2]['Q_prim']
                solver_param['hyper']     = False


        else:

            solver_param, state, rom_param = solver.advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        # update the plot
        if solver_param['visual'] and (iter % solver_param['vis_update_interval'] == 0):

            visual_func.in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param)

        print('Iteration: ' + str(iter))

        state['time'] = state['time'] + solver_param['dt']

        iter = iter + 1

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    print('Simulation successfully completed!')

    return state


