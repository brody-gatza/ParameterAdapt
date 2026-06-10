import os
import numpy as np
import matplotlib.pyplot as plt

def visual_var_collector(solver_param):

    visual_options            = ['Density','Velocity','Pressure','Temprature','MF','Heat Release']
    visual_options_indx_stack = [   0     ,     1    ,     2    ,      3     , 4  ,      -1      ]
    
    for item in visual_options:

        if solver_param['variable1'] == item:

            indx = visual_options.index(item)
            visual_var1 = visual_options_indx_stack[indx]

        elif solver_param['variable2'] == item:

            indx = visual_options.index(item)
            visual_var2 = visual_options_indx_stack[indx]

        elif solver_param['variable3'] == item:

            indx = visual_options.index(item)
            visual_var3 = visual_options_indx_stack[indx]

        elif solver_param['variable4'] == item:

            indx = visual_options.index(item)
            visual_var4 = visual_options_indx_stack[indx]

    if solver_param['variable1'] == 'None':
        
        visual_var1 = []

    if solver_param['variable2'] == 'None':
        
        visual_var2 = []

    if solver_param['variable3'] == 'None':
        
        visual_var3 = []       

    if solver_param['variable4'] == 'None':
        
        visual_var4 = []

    visual_param = {}

    visual_param['visual_var1'] = visual_var1
    visual_param['visual_var2'] = visual_var2
    visual_param['visual_var3'] = visual_var3
    visual_param['visual_var4'] = visual_var4

    return visual_param

def initial_plot(axs,visual_param):

    # Plot in the first subplot (top-left)
    plot1 , = axs[0, 0].plot([], [] , color = 'C0' ,linestyle='-')

    # Plot in the second subplot (top-right)
    plot2 , = axs[0, 1].plot([], [], color='C1', linestyle='-')
    
    # Plot in the third subplot (bottom-left)
    plot3 , = axs[1, 0].plot([], [], color='C2', linestyle='-')
   
    # Plot in the fourth subplot (bottom-right)
    plot4 , = axs[1, 1].plot([], [], color='C3', linestyle='-')

    visual_param['plot1'] = plot1 
    visual_param['plot2'] = plot2 
    visual_param['plot3'] = plot3 
    visual_param['plot4'] = plot4

    scatter1 = axs[0, 0].scatter([], [] , 7 ,color='black',marker='o')
    scatter2 = axs[0, 1].scatter([], [] , 7 ,color='black',marker='o') 
    scatter3 = axs[1, 0].scatter([], [] , 7 ,color='black',marker='o')
    scatter4 = axs[1, 1].scatter([], [] , 7 ,color='black',marker='o')

    visual_param['scatter1'] = scatter1
    visual_param['scatter2'] = scatter2
    visual_param['scatter3'] = scatter3
    visual_param['scatter4'] = scatter4
    
    return visual_param

def in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param):

    prim_results = state['prim_results_save']

    x = solver_param['x']

    visual_var1 = visual_param['visual_var1'] 
    visual_var2 = visual_param['visual_var2'] 
    visual_var3 = visual_param['visual_var3'] 
    visual_var4 = visual_param['visual_var4'] 

    y11= prim_results[visual_var1,:]
    y21= prim_results[visual_var2,:]
    y31= prim_results[visual_var3,:]
    y41= prim_results[visual_var4,:]


    if y11.size == 0 : 

        y11 = 0 * prim_results[0,:]

    if y21.size == 0 : 

        y21 = 0 * prim_results[0,:]

    if y31.size == 0 : 

        y31 = 0 * prim_results[0,:]

    if y41.size == 0 : 

        y41 = 0 * prim_results[0,:]

    plot11 = visual_param['plot1']
    plot21 = visual_param['plot2']
    plot31 = visual_param['plot3']
    plot41 = visual_param['plot4']

    plot11.set_data(x,y11)
    plot21.set_data(x,y21)
    plot31.set_data(x,y31)
    plot41.set_data(x,y41)

    axs[0,0].set_ylabel(solver_param['variable1'])
    axs[0,1].set_ylabel(solver_param['variable2'])
    axs[1,0].set_ylabel(solver_param['variable3'])
    axs[1,1].set_ylabel(solver_param['variable4'])

    axs[0,0].relim()
    axs[0,0].autoscale()

    axs[0,1].relim()
    axs[0,1].autoscale()

    axs[1,0].relim()
    axs[1,0].autoscale()

    axs[1,1].relim()
    axs[1,1].autoscale()

    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        scatter1 = visual_param['scatter1']
        scatter2 = visual_param['scatter2']
        scatter3 = visual_param['scatter3']
        scatter4 = visual_param['scatter4']

        scatter1.set_offsets(np.column_stack((x[S_indx_user], y11[S_indx_user])))
        scatter2.set_offsets(np.column_stack((x[S_indx_user], y21[S_indx_user])))
        scatter3.set_offsets(np.column_stack((x[S_indx_user], y31[S_indx_user])))
        scatter4.set_offsets(np.column_stack((x[S_indx_user], y41[S_indx_user])))

    plt.pause(1e-6)

    if solver_param['save_visual']:

        dir_results = os.path.join(solver_param['dir_results'])
        iter        = solver_param['iter']
        save_title  = str(iter)+'iteration'

        plt.savefig((os.path.join(dir_results, 'plots'     , f"{save_title}_plot.png")) )