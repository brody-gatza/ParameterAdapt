import non_linear_terms
import numpy as np
import scipy.optimize 

def advance_time(solver_param,state):
    # time integration
    if solver_param['time_scheme'] == 'Explicit - FD Euler':

        state   = explicit_fd_euler(solver_param,state)

    # elif solver_param['time_scheme'] == 'Implicit - BD Euler':

    #     mass , momx , energy  = implicit_bd_euler(mass,momx,energy,dx,dt,gamma,vol,res_tol,slope_limiter)
    return state

def explicit_fd_euler( solver_param,state ):

    dt    = solver_param['dt']
    Q_old = state['Q_cons']
    dQ_dt = state['d_flux_dx'] 

    d_Q   = dt * dQ_dt 
    Q_new = Q_old + d_Q

    state['Q_cons'] = Q_new

    return state
