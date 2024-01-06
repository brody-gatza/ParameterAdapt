def visual_var_collector(solver_param):

    visual_options            = ['Density','Pressure','Temprature','Velocity','Heat Release','Mass Fraction']
    visual_options_indx_stack = [   0     ,     2    ,      0     ,     1    ,      0       ,       0       ]
    
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

    return visual_var1 , visual_var2 , visual_var3 , visual_var4


def initial_plot(axs):
    plots = {}
    # Plot in the first subplot (top-left)
    plot11 , = axs[0, 0].plot([], [] , color = 'C0' ,linestyle='-'  , label = 'ROM')
    plot12 , = axs[0, 0].plot([], [] , color = 'C0' ,linestyle='--' , label = 'FOM')
    # plots['scatter1'] = axs[0, 0].scatter([], [],color='black',marker='o')
    # axs[0, 0].set_title('var1')

    # Plot in the second subplot (top-right)
    plot21 , = axs[0, 1].plot([], [], color='C1', linestyle='-'  ,  label = 'ROM')
    plot22 , = axs[0, 1].plot([], [], color='C1', linestyle='--' ,  label = 'FOM')
    # axs[0, 1].set_title('var2')

    # Plot in the third subplot (bottom-left)
    plot31 , = axs[1, 0].plot([], [], color='C2', linestyle='-' , label = 'ROM')
    plot32 , = axs[1, 0].plot([], [], color='C2', linestyle='--', label = 'FOM')
    # axs[1, 0].set_title('var3')

    # Plot in the fourth subplot (bottom-right)
    plot41 , = axs[1, 1].plot([], [], color='C3', linestyle='-' , label = 'ROM')
    plot42 , = axs[1, 1].plot([], [], color='C3', linestyle='--', label = 'FOM')
    # axs[1, 1].set_title('var4')

    plots['plot11'] = plot11 
    plots['plot21'] = plot21 
    plots['plot31'] = plot31 
    plots['plot41'] = plot41

    plots['plot12'] = plot12 
    plots['plot22'] = plot22 
    plots['plot32'] = plot32 
    plots['plot42'] = plot42 
    

    return plots

def in_progress_plot(fig,axs,x,prim_results,plots,visual_var1,visual_var2,visual_var3,visual_var4,solver_param,iter,training_data_prim,rom_flag):
    

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

    plot11 = plots['plot11']
    plot21 = plots['plot21']
    plot31 = plots['plot31']
    plot41 = plots['plot41']

    plot11.set_data(x,y11)
    plot21.set_data(x,y21)
    plot31.set_data(x,y31)
    plot41.set_data(x,y41)

    axs[0,0].set_ylim(min(y11,default=0)*0.0099 , max(y11,default=0)*1.1)
    axs[0,1].set_ylim(min(y21,default=0)*0.0099 , max(y21,default=0)*1.1)
    axs[1,0].set_ylim(min(y31,default=0)*0.0099 , max(y31,default=0)*1.1)
    axs[1,1].set_ylim(min(y41,default=0)*0.0099 , max(y41,default=0)*1.1)

    
    # axs[0,0].set_ylim(0 , 6)
    # axs[0,1].set_ylim(970000 , 987500)
    # axs[1,0].set_ylim(0 , 8)
    # axs[1,1].set_ylim(min(y4,default=0)*0.0099 , max(y4,default=0)*1.001)

    axs[0,0].set_xlim( 0 , x[-2]*1.05)
    axs[0,1].set_xlim( 0 , x[-2]*1.05)
    axs[1,0].set_xlim( 0 , x[-2]*1.05)
    axs[1,1].set_xlim( 0 , x[-2]*1.05)

    axs[0,0].set_ylabel(solver_param['variable1'])
    axs[0,1].set_ylabel(solver_param['variable2'])
    axs[1,0].set_ylabel(solver_param['variable3'])
    axs[1,1].set_ylabel(solver_param['variable4'])



    if rom_flag == True :

        y12= training_data_prim[visual_var1,:,iter]
        y22= training_data_prim[visual_var2,:,iter]
        y32= training_data_prim[visual_var3,:,iter]
        y42= training_data_prim[visual_var4,:,iter]

        if y12.size == 0 : 

            y12 = 0 * prim_results[0,:]

        if y22.size == 0 : 

            y22 = 0 * prim_results[0,:]

        if y32.size == 0 : 

            y32 = 0 * prim_results[0,:]

        if y42.size == 0 : 

            y42 = 0 * prim_results[0,:]

        plot12 = plots['plot12']
        plot22 = plots['plot22']
        plot32 = plots['plot32']
        plot42 = plots['plot42']

        plot12.set_data(x,y12)
        plot22.set_data(x,y22)
        plot32.set_data(x,y32)
        plot42.set_data(x,y42)

        axs[0,0].legend()
        axs[0,1].legend()
        axs[1,0].legend()
        axs[1,1].legend()


    if iter % 1000 == 0:

        fig.canvas.draw()
        fig.canvas.flush_events()
        



    



