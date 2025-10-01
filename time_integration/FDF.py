def advance_time(solver_param,rom_param,state):

    dt    = solver_param['dt']
    Q_old = state['Q_cons']
    dQ_dt = state['d_flux_dx']
    d_Q   = dt * dQ_dt 
    Q_new = Q_old + d_Q

    state['Q_cons'] = Q_new

    return state