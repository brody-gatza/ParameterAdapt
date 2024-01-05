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

    # Plot in the first subplot (top-left)
    plot1 , = axs[0, 0].plot([], [])
    scatter1 = axs[0, 0].scatter([], [],color='black',marker='o')
    # axs[0, 0].set_title('var1')

    # Plot in the second subplot (top-right)
    plot2 , = axs[0, 1].plot([], [], color='orange')
    # axs[0, 1].set_title('var2')

    # Plot in the third subplot (bottom-left)
    plot3 , = axs[1, 0].plot([], [], color='green')
    # axs[1, 0].set_title('var3')

    # Plot in the fourth subplot (bottom-right)
    plot4 , = axs[1, 1].plot([], [], color='red')
    # axs[1, 1].set_title('var4')

    return plot1 , plot2 , plot3 , plot4  , scatter1

def in_progress_plot(fig,axs,x,FOM_prim_results,plot1,plot2,plot3,plot4,visual_var1,visual_var2,visual_var3,visual_var4,solver_param,iter):
    

    y1= FOM_prim_results[visual_var1,:]
    y2= FOM_prim_results[visual_var2,:]
    y3= FOM_prim_results[visual_var3,:]
    y4= FOM_prim_results[visual_var4,:]

    if y1.size == 0 : 

        y1 = 0 * FOM_prim_results[0,:]

    if y2.size == 0 : 

        y2 = 0 * FOM_prim_results[0,:]

    if y3.size == 0 : 

        y3 = 0 * FOM_prim_results[0,:]

    if y4.size == 0 : 

        y4 = 0 * FOM_prim_results[0,:]


    plot1.set_data(x,y1)
    plot2.set_data(x,y2)
    plot3.set_data(x,y3)
    plot4.set_data(x,y4)

    axs[0,0].set_ylim(min(y1,default=0)*0.0099 , max(y1,default=0)*1.1)
    # axs[0,0].set_ylim(0.302 , 0.303)
    axs[0,1].set_ylim(min(y2,default=0)*0.0099 , max(y2,default=0)*1.1)
    axs[1,0].set_ylim(min(y3,default=0)*0.0099 , max(y3,default=0)*1.1)
    axs[1,1].set_ylim(min(y4,default=0)*0.0099 , max(y4,default=0)*1.1)

    
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



    if iter % 50 == 0:

        fig.canvas.draw()
        fig.canvas.flush_events()
        



    



