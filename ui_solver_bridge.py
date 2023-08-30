import solver_functions
import time_integrator_functions
import visualization_functions
import numpy as np
import matplotlib.pyplot as plt


def driver(self):
    
    # collect all of variables from user interface
    solver_param = solver_functions.solver_parameters_collector(self)
    slope_limiter = True
    
    # some basic parameters
    dx  = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    vol = dx
    dt = solver_param['dt']
    x  =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )

    # get the initial condition
    rho , vx , p = solver_functions.ic_generator(solver_param)

    # add ghost cells
    rho , vx , p = solver_functions.add_ghost_cell(rho , vx , p)

    # get the gas properties
    gamma = 1.4

    # convert prim to cons
    mass , momx , energy = solver_functions.prim2cons_converter(rho , vx , p , gamma , vol)

    # stack cons and prim variables
    full_FOM_cons_results      = np.zeros(( 3 , int(solver_param['cell_number'])+2  )  )
    FOM_cons_results           = np.zeros(( 3 , int(solver_param['cell_number'])    )  )
    full_FOM_cons_results[0,:] = mass 
    full_FOM_cons_results[1,:] = momx 
    full_FOM_cons_results[2,:] = energy

    full_FOM_prim_results      = np.zeros(( 3 , int(solver_param['cell_number'])+2  )  )
    FOM_prim_results           = np.zeros(( 3 , int(solver_param['cell_number'])    )  )
    full_FOM_prim_results[0,:] = rho 
    full_FOM_prim_results[1,:] = vx 
    full_FOM_prim_results[2,:] = p

    # create plot
    fig , axs = plt.subplots(2,2)
    visual_var1,visual_var2,visual_var3,visual_var4 = visualization_functions.visual_var_collector(solver_param)
    plot1 , plot2 , plot3 , plot4 = visualization_functions.initial_plot(axs)
    
    # begin simulation

    for iter in range(solver_param['num_step']):

        # convert cons to prim
        rho , vx , p = solver_functions.cons2prim_converter(mass , momx , energy , gamma , vol)

        # calculate gradients
        d_rho_dx = solver_functions.gradient_calculator(rho , dx)
        d_vx_dx  = solver_functions.gradient_calculator(vx , dx)
        d_p_dx   = solver_functions.gradient_calculator(p , dx)

        if slope_limiter:

            d_rho_dx = solver_functions.slope_limit(rho , dx , d_rho_dx)
            d_vx_dx  = solver_functions.slope_limit(vx  , dx , d_vx_dx )
            d_p_dx   = solver_functions.slope_limit(p   , dx , d_p_dx  )

        # extrapolate from center to face
        rho_face_left , rho_face_right = solver_functions.extrapolate_center2face(rho,d_rho_dx,dx)
        vx_face_left  , vx_face_right  = solver_functions.extrapolate_center2face(vx,d_vx_dx,dx)
        p_face_left   , p_face_right   = solver_functions.extrapolate_center2face(p,d_p_dx,dx)

        # compute flux
        flux_mass, flux_momx, flux_energy = solver_functions.flux_calculator(rho_face_left , rho_face_right , vx_face_left , vx_face_right , p_face_left , p_face_right , gamma)

        # inviscid flux vector terms
        dflux_mass_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_mass   , dx )
        dflux_momx_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_momx   , dx )
        dflux_energy_dx = solver_functions.inviscid_d_flux_dx_calculator( flux_energy , dx )

        # apply flux terms
        d_mass_dt   = - dflux_mass_dx   * dx
        d_momx_dt   = - dflux_momx_dx   * dx
        d_energy_dt = - dflux_energy_dx * dx

        # time integration
        
        mass   = time_integrator_functions.explicit_rk4(mass   , d_mass_dt   , dt)
        momx   = time_integrator_functions.explicit_rk4(momx   , d_momx_dt   , dt)
        energy = time_integrator_functions.explicit_rk4(energy , d_energy_dt , dt)

        # convert cons to prim
        rho , vx , p = solver_functions.cons2prim_converter(mass , momx , energy , gamma , vol)

        # update results
        full_FOM_cons_results[0,:] = mass 
        full_FOM_cons_results[1,:] = momx 
        full_FOM_cons_results[2,:] = energy

        full_FOM_prim_results[0,:] = rho
        full_FOM_prim_results[1,:] = vx 
        full_FOM_prim_results[2,:] = p

        # update results without ghost cells
        FOM_cons_results[0,:] = mass  [1:-1]
        FOM_cons_results[1,:] = momx  [1:-1]
        FOM_cons_results[2,:] = energy[1:-1]

        FOM_prim_results[0,:] = rho   [1:-1]
        FOM_prim_results[1,:] = vx    [1:-1]
        FOM_prim_results[2,:] = p     [1:-1]

        # visualization
        visualization_functions.in_progress_plot(fig,axs,x,FOM_prim_results,plot1,plot2,plot3,plot4,visual_var1,visual_var2,visual_var3,visual_var4,solver_param,iter)
        plt.show(block=False)
        print('Iteration: ' + str(iter))






        







        


        
