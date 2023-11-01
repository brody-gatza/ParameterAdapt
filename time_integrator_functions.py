import non_linear_terms
import numpy as np
import scipy.optimize 

def explicit_fd_euler( var , ode , dt ):

    d_var = dt * ode 
    var =  var + d_var

    return var

def implicit_bd_euler (mass,momx,energy,dx,dt,gamma,vol,res_tol,slope_limiter):

    def cost_fun(q,dx,gamma,vol,slope_limiter):

        mass_guess      = q[0]
        momx_guess      = q[1]
        energy_guess    = q[2]

        dflux_mass_dx , dflux_momx_dx , dflux_energy_dx = non_linear_terms.flux_calculator(mass_guess,momx_guess,energy_guess,dx,gamma,vol,slope_limiter)

        # apply flux terms
        d_mass_dt   = - dflux_mass_dx   * dx
        d_momx_dt   = - dflux_momx_dx   * dx
        d_energy_dt = - dflux_energy_dx * dx

        cost_mass   = mass_guess   - mass   - (dt * d_mass_dt)
        cost_momx   = momx_guess   - momx   - (dt * d_momx_dt)
        cost_energy = energy_guess - energy - (dt * d_energy_dt)

        cost = np.vstack((cost_mass,cost_momx,cost_energy))
        cost = np.sum(cost**2)

        return cost
    
    q_old = np.array((mass,momx,energy)).flatten()
    q_0   = q_old
    q_new =  scipy.optimize.fsolve(cost_fun,q_0,args=(dx,gamma,vol,slope_limiter),xtol=res_tol)

    mass   = q_new[0]
    momx   = q_new[1]
    energy = q_new[2]

    return mass , momx , energy

    # mass_old         = mass
    # momx_old         = momx
    # energy_old       = energy

    # mass_guess       = mass_old
    # momx_guess       = momx_old
    # energy_guess     = energy_old
    
    # counter          = 1

    # while error > res_tol:

    #     if counter == 1:

    #         mass_new   = mass_guess
    #         momx_new   = momx_guess
    #         energy_new = energy_guess

    # ### calculating the right-hand-side of governing equation (identical to  the one we have in main solver) ###

    #     mass_function_old   = mass_new   - mass_old     - dt * rhs_mass
    #     momx_function_old   = momx_new   - momx_old     - dt * rhs_momx
    #     energy_function_old = energy_new - energy_old   - dt * rhs_energy

    #     dflux_mass_dx , dflux_momx_dx , dflux_energy_dx = non_linear_terms.flux_calculator(mass,momx,energy,dx,gamma,vol,slope_limiter)

    #     # apply flux terms
    #     d_mass_dt   = - dflux_mass_dx   * dx
    #     d_momx_dt   = - dflux_momx_dx   * dx
    #     d_energy_dt = - dflux_energy_dx * dx

    #     ###  end of calculating the right-hand-side of governing equation ###

    #     rhs_mass        = d_mass_dt   
    #     rhs_momx        = d_momx_dt   
    #     rhs_energy      = d_energy_dt 

    #     # cost_functions

    #     mass_function_new   = mass_new   - mass_old     - (dt * rhs_mass)
    #     momx_function_new   = momx_new   - momx_old     - (dt * rhs_momx)
    #     energy_function_new = energy_new - energy_old   - (dt * rhs_energy)

    #     jacobian_mass_function   = (mass_function_new   - mass_function_old)   / dt
    #     jacobian_momx_function   = (momx_function_new   - momx_function_old)   / dt
    #     jacobian_energy_function = (energy_function_new - energy_function_old) / dt

    #     mass_new   = mass_old   - (mass_function_old / jacobian_mass_function)
    #     momx_new   = momx_old   - (momx_function_old / jacobian_momx_function)
    #     energy_new = energy_old - (energy_function_old / jacobian_energy_function)

    #     q_old      = np.vstack((mass_old,momx_old,energy_old))
    #     q_new      = np.vstack((mass_new,momx_new,energy_new))
        
    #     error = np.linalg.norm(q_new - q_old) / np.linalg.norm(q_old)

    #     counter =  counter + 1

    # mass    = mass_new
    # momx    = momx_new
    # energy  = energy_new
    
    # return mass , momx , energy