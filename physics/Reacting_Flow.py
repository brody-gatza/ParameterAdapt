import numpy as np
import cantera as ct
from scipy.interpolate import interp1d


from utils import reshape_func
from boundary_condition import bc_func
 
def prim2cons_converter(solver_param, state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']

    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    vol = solver_param['vol']

    rho = Q_prim_user[0,:]
    vx  = Q_prim_user[1,:] 
    P   = Q_prim_user[2,:]
    T   = Q_prim_user[3,:]
    Y   = Q_prim_user[4:,:]

    Y_cantera = reshape_func.find_mass_fraction_full_cantera(Y)
    
    state['gas_array'].TPY = T,P,Y_cantera

    internal_energy = np.squeeze(state['gas_array'].int_energy_mass)

    internal_energy_tot = internal_energy + (0.5 * (vx **2))

    mass   = rho * vol
    momx   = rho * vx * vol
    energy = (rho * internal_energy_tot) * vol
    mass_species = rho * Y * vol

    state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()

    return state

def cons2prim_converter(solver_param, state):

    vol         = solver_param['vol']

    Q_cons      = state['Q_cons']
    Q_cons_user = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    mass        = Q_cons_user[0,:]
    momx        = Q_cons_user[1,:]
    energy      = Q_cons_user[2,:]
    mass_species= Q_cons_user[3:,:]
    
    rho = mass / vol
    vx  = momx / rho / vol
    MF  = mass_species / rho / vol

    MF_ct = reshape_func.find_mass_fraction_full_cantera(MF)

    sp_vol = 1/rho

    internal_energy = (energy/vol/rho)-(0.5*vx**2)

    try:

        state['gas_array'].UVY = internal_energy,sp_vol,MF_ct

        T = np.squeeze(state['gas_array'].T)
        P = np.squeeze(state['gas_array'].P)

    except:

        # sometimes rom can give shitty results but it might recover in the later time steps
        # but when cantera gets invalid states directly raises an error and stops the whole simulation
        # to avoid this issue I implemented this sloppy way of finding prim vars 
        # I am just finding a coefficient (psudo Cp, Cv) that converts energy term to pressure and temperature 
        # by looking at the previous time step solution and finally I trim the invalid results by max and min pressure

        Q_prim   = state['Q_prim']

        Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],
                                                                solver_param['cell_number'],
                                                                Q_prim)
        
        rho_spare = Q_prim_user[0,:]
        vx_spare  = Q_prim_user[1,:]
        P_spare   = Q_prim_user[2,:]
        T_spare   = Q_prim_user[3,:]
        Y_spare   = Q_prim_user[4:,:]
        Y_spare_ct= reshape_func.find_mass_fraction_full_cantera(Y_spare)

        state['gas_array'].TPY = T_spare, P_spare, Y_spare_ct

        coef_temp  = np.squeeze(state['gas_array'].int_energy_mass)/T_spare
        coef_press = np.squeeze(state['gas_array'].int_energy_mass)/P_spare

        T = np.abs(internal_energy / coef_temp)
        P = np.abs(internal_energy / coef_press)

        Q_prim = np.vstack((rho,vx,P,T,MF))

        P_max = np.max(P_spare)
        P_min = np.min(P_spare)

        P_max_indx = np.argmax(P_spare)
        P_min_indx = np.argmin(P_spare)

        indx_pass_max = np.where(Q_prim[2,:]>P_max)[0]
        indx_pass_min = np.where(Q_prim[2,:]<P_min)[0]

        Q_prim[:,indx_pass_max] = Q_prim_user[:,P_max_indx].reshape(-1,1)
        Q_prim[:,indx_pass_min] = Q_prim_user[:,P_min_indx].reshape(-1,1)

        rho = Q_prim_user[0,:]
        vx  = Q_prim_user[1,:]
        P   = Q_prim_user[2,:]
        T   = Q_prim_user[3,:]
        Y   = Q_prim_user[4:,:]

        Y_ct= reshape_func.find_mass_fraction_full_cantera(Y)

        state['gas_array'].TPY = T, P, Y_ct

        internal_energy = np.squeeze(state['gas_array'].int_energy_mass)

        internal_energy_tot = internal_energy + (0.5 * (vx **2))

        mass   = rho * vol
        momx   = rho * vx * vol
        energy = (rho * internal_energy_tot) * vol
        mass_species = rho * Y * vol

        state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()

    state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

    return state

def first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):
    
    state            = cons2prim_converter(solver_param,state)
    state            = bc_func.update_ghost_cell(solver_param,state)
    state            = prim2cons_converter(solver_param,state)

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]


    Y      = Q_prim_user[4:,:]
    Y_ct   = reshape_func.find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    state['gas_array'].TPY = temp,press,Y_ct

    c       = np.squeeze(state['gas_array'].sound_speed)
    int_en  = np.squeeze(state['gas_array'].int_energy_mass)
    h       = np.squeeze(state['gas_array'].enthalpy_mass)

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))
    
    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        S_indx_user      = S_indx_user + 2
        range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
        range_flux_left  = range_flux_right - 1

        rho_left  = rho[range_flux_left]
        rho_right = rho[range_flux_right]

        vx_left  = vx[range_flux_left]
        vx_right = vx[range_flux_right]

        htot_left  = htot[range_flux_left]
        htot_right = htot[range_flux_right]

        press_left  = press[range_flux_left]
        press_right = press[range_flux_right]

        temp_left  = temp[range_flux_left]
        temp_right = temp[range_flux_right]

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        int_en_left    = int_en[range_flux_left]
        int_en_right   = int_en[range_flux_right]

        Y_left         = Y[:,range_flux_left]
        Y_right        = Y[:,range_flux_right]

        del_q_prim = np.zeros((2*len(S_indx_user),num_state_var,1))

        diss_matrix = np.zeros((2*len(S_indx_user),num_state_var,num_state_var))

        gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        temp_left  = temp[:-1]
        temp_right = temp[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        int_en_left  = int_en[:-1]
        int_en_right = int_en[1:]

        Y_left         = Y[:,:-1]
        Y_right        = Y[:,1:]

        del_q_prim = np.zeros((cell_num-1,num_state_var,1))

        diss_matrix = np.zeros((cell_num-1,num_state_var,num_state_var))

        gas_array = ct.SolutionArray(state['gas'],cell_num-1)
    
    R   = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
    Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    emoy=(R*int_en_right+int_en_left)/(R+1)              
    hmoy=Hmoy - (0.5*umoy*umoy)
    Ymoy=(R*Y_right+Y_left)/(R+1)                  

    Ymoy_ct = reshape_func.find_mass_fraction_full_cantera(Ymoy)

    gas_array.UVY = emoy,1/rmoy,Ymoy_ct

    Pmoy         = gas_array.P
    Tmoy         = gas_array.T
    cpmoy        = gas_array.cp
    cmoy         = gas_array.sound_speed
    meanMWmoy    = gas_array.mean_molecular_weight
    MWmoy        = gas_array.molecular_weights
    partial_hmoy = gas_array.partial_molar_enthalpies / MWmoy
        
    d_rho_d_press = rmoy / Pmoy
    d_rho_d_temp = -rmoy/Tmoy

    # d_rho_d_mass_frac = np.zeros_like(Ymoy)

    #     for sp in range(len(Ymoy)):

    #         d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])
    d_rho_d_mass_frac_const_term =  (1/MWmoy[-1] - 1/MWmoy[:-1])
    d_rho_d_mass_frac = rmoy * meanMWmoy * d_rho_d_mass_frac_const_term[:,np.newaxis]

    d_enth_d_press = 0
    d_enth_d_temp  = cpmoy

    # d_enth_d_mass_frac = np.zeros_like(Ymoy)

    # for sp in range(len(Ymoy)):
    
    d_enth_d_mass_frac = partial_hmoy[:,:-1] - partial_hmoy[:,-1,np.newaxis]

    #     # Gamma terms for energy equation
    g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
    g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
    g_mass_frac = rmoy[:,np.newaxis] * d_enth_d_mass_frac + Hmoy[:,np.newaxis] * d_rho_d_mass_frac.T

    #     # Characteristic speeds
    lambda1 = umoy + cmoy
    lambda2 = umoy - cmoy
    lambda1_abs = np.absolute(lambda1)
    lambda2_abs = np.absolute(lambda2)

    r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
    alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
    beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
    phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

    eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
    psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

    vel_abs = np.absolute(umoy)

    beta_star = beta * psi
    beta_e = beta * (rmoy * g_press + g_temp * eta)
    phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
    phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
    m = rmoy * alpha
    e = rmoy * umoy * alpha

    delta_p = press_left-press_right
    delta_u = vx_left-vx_right
    delta_T = temp_left-temp_right
    delta_Y = (Y_left-Y_right)

    # del_q_prim = np.zeros((cell_num-1,num_state_var,1))

    del_q_prim[:,0,:]  = delta_p.reshape(-1,1)
    del_q_prim[:,1,:]  = delta_u.reshape(-1,1)
    del_q_prim[:,2,:]  = delta_T.reshape(-1,1)
    del_q_prim[:,3:,:] = delta_Y.T[:,:,np.newaxis]


    diss_matrix[: , 0 , 0 ] = phi_star
    diss_matrix[: , 0 , 1 ] = beta_star
    diss_matrix[: , 0 , 2 ] = vel_abs * d_rho_d_temp
    diss_matrix[: , 0 , 3:] = vel_abs[:,np.newaxis] * d_rho_d_mass_frac.T

    diss_matrix[: , 1 , 0 ] = umoy * phi_star + r_roe
    diss_matrix[: , 1 , 1 ] = umoy * beta_star + m
    diss_matrix[: , 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
    diss_matrix[: , 1 , 3:] = (umoy * vel_abs)[:,np.newaxis] * d_rho_d_mass_frac.T

    diss_matrix[: , 2 , 0 ] = phi_e + r_roe * umoy
    diss_matrix[: , 2 , 1 ] = beta_e + e
    diss_matrix[: , 2 , 2 ] = g_temp * vel_abs
    diss_matrix[: , 2 , 3:] = g_mass_frac * vel_abs[:,np.newaxis]

    diss_matrix[: , 3:, 0] = Ymoy.T * phi_star[:,np.newaxis]
    diss_matrix[: , 3:, 1] = Ymoy.T * beta_star[:,np.newaxis]
    diss_matrix[: , 3:, 2] = Ymoy.T * (vel_abs * d_rho_d_temp)[:,np.newaxis]


    idx_out, idx_in = np.meshgrid(np.arange(num_state_var-3), 
                                np.arange(num_state_var-3), indexing='ij')
    
    diag_mask = idx_out == idx_in

    off_diag = vel_abs[:,np.newaxis] * Ymoy.T * d_rho_d_mass_frac.T
    off_diag = np.repeat(off_diag[:,:,np.newaxis],np.shape(off_diag)[1],axis=2)

    diag = vel_abs[:,np.newaxis] * (rmoy[:,np.newaxis] + Ymoy.T * d_rho_d_mass_frac.T)

    diss_matrix[:,3:,3:] = off_diag
    diss_matrix[:,3:,3:][:,diag_mask] = diag

    dissipation = np.matmul(diss_matrix , del_q_prim)

    if solver_param['hyper'] == True:

        flux[0 ,range_flux_right] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
        flux[1 ,range_flux_right] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
        flux[2 ,range_flux_right] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
        flux[3:,range_flux_right] = 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T


    else:

        flux[0,1:-1] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
        flux[1,1:-1] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
        flux[2,1:-1] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
        flux[3:,1:-1]= 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T
            

    state['flux_cons'] = flux

    return state

def first_order_roe_viscous_flux_calculator(solver_param,rom_param,state):

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = reshape_func.find_mass_fraction_full(Y)
    Y_ct   = reshape_func.find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    state['gas_array'].TPY = temp,press,Y_ct

    c       = np.squeeze(state['gas_array'].sound_speed)
    int_en  = np.squeeze(state['gas_array'].int_energy_mass)
    h       = np.squeeze(state['gas_array'].enthalpy_mass)

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        S_indx_user      = S_indx_user + 2
        range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
        range_flux_left  = range_flux_right - 1

        rho_left  = rho[range_flux_left]
        rho_right = rho[range_flux_right]

        vx_left  = vx[range_flux_left]
        vx_right = vx[range_flux_right]

        htot_left  = htot[range_flux_left]
        htot_right = htot[range_flux_right]

        press_left  = press[range_flux_left]
        press_right = press[range_flux_right]

        temp_left  = temp[range_flux_left]
        temp_right = temp[range_flux_right]

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        int_en_left    = int_en[range_flux_left]
        int_en_right   = int_en[range_flux_right]

        Y_left         = Y[:,range_flux_left]
        Y_right        = Y[:,range_flux_right]

        Y_full_left         = Y_full[:,range_flux_left]
        Y_full_right        = Y_full[:,range_flux_right]

        left_state = Q_cons_user[:,range_flux_left]
        right_state= Q_cons_user[:,range_flux_right]  
   
        gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))   

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        temp_left  = temp[:-1]
        temp_right = temp[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        int_en_left  = int_en[:-1]
        int_en_right = int_en[1:]

        Y_left         = Y[:,:-1]
        Y_right        = Y[:,1:]

        Y_full_left         = Y_full[:,:-1]
        Y_full_right        = Y_full[:,1:]

        left_state = Q_cons_user[:,:-1]
        right_state= Q_cons_user[:,1:]

        gas_array = ct.SolutionArray(state['gas'],cell_num-1)


    R = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
    Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    emoy=(R*int_en_right+int_en_left)/(R+1)              
    hmoy=Hmoy - (0.5*umoy*umoy)
    Ymoy=(R*Y_right+Y_left)/(R+1)                  

    Ymoy_ct = reshape_func.find_mass_fraction_full_cantera(Ymoy)

    gas_array.UVY = emoy,1/rmoy,Ymoy_ct

    dyn_vsc_mix                 = gas_array.viscosity
    mass_diff_mix               = gas_array.mix_diff_coeffs_mass
    therm_cond_mix              = gas_array.thermal_conductivity

    MW                          = gas_array.molecular_weights
    MW_mix                      = gas_array.mean_molecular_weight
    enthalpies                  = gas_array.partial_molar_enthalpies / MW


    du_dx = (vx_right - vx_left) /dx
    dT_dx = (temp_right - temp_left) /dx
    dY_dx = (Y_full_right - Y_full_left) / dx

    corr_vel  = np.sum(mass_diff_mix*dY_dx.T,axis=1)

    diff_vel  = -mass_diff_mix*dY_dx.T/Ymoy_ct[0,:,:] + corr_vel[:,np.newaxis]

    tau       = 4/3 * dyn_vsc_mix * du_dx

    q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies*diff_vel*Ymoy_ct[0,:,:],axis=1)

    if solver_param['hyper'] == True:

        flux[0 ,range_flux_right] = 0 
        flux[1 ,range_flux_right] = tau
        flux[2 ,range_flux_right] = (umoy * tau) - q
        flux[3:,range_flux_right] = (rmoy[:,np.newaxis]*mass_diff_mix[:,:-1]*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

    else: 

        flux[0 ,1:-1] = 0 
        flux[1 ,1:-1] = tau
        flux[2 ,1:-1] = (umoy * tau) - q
        flux[3:,1:-1] = (rmoy[:,np.newaxis]*mass_diff_mix[:,:-1]*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

        
    state['flux_visc_cons'] = flux

    return state

def d_flux_dx_calculator(solver_param,rom_param,state):

    # apply viscous flux if exsted
    if solver_param['viscous_flag']:
        
        flux = state['flux_cons']-state['flux_visc_cons']

    else:

        flux = state['flux_cons']
    
    # apply the flux 
    if solver_param['hyper'] == True:

        S_indx_user      = rom_param['S_indx_user']
        S_indx_user      = S_indx_user + 2
        range_flux_right = S_indx_user + 1
        range_flux_left  = S_indx_user

        d_flux_dx = -(flux[:, range_flux_right] - flux[:, range_flux_left])

    else:

        d_flux_dx = -(flux[:, 1:] - flux[:, :-1])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def injection_correction(solver_param,state):

    # read current states
    Q_prim      = state['Q_prim']
    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    Q_cons      = state['Q_cons']
    Q_cons_user = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    rho         = Q_prim_user[0,:]
    u           = Q_prim_user[1,:]    
    P           = Q_prim_user[2,:]
    T           = Q_prim_user[3,:]
    Y           = Q_prim_user[4:,:]
    Y_ct        = reshape_func.find_mass_fraction_full_cantera(Y)

    vol         = solver_param['vol']

    # read injection regions index
    add_indx    = state['injection_add_final']
    sub_indx    = state['injection_sub_init'] 

    indx_init   = np.argmax(P) - sub_indx
    indx_final  = indx_init + add_indx
    detonation  = np.arange(indx_init,indx_final)%(solver_param['cell_number']+4)
    inj_indx    = np.arange(0,solver_param['cell_number']+4,1)
    inj_indx    = inj_indx[~np.isin(inj_indx,detonation)]

    # create mixed gas state
    gas_array_current  = ct.SolutionArray(state['gas'],(1,len(inj_indx)))
    gas_array_inject   = ct.SolutionArray(state['gas'],(1,len(inj_indx)))
    gas_array_mix      = ct.SolutionArray(state['gas'],(1,len(inj_indx)))

    # compute injection rate 
    rho_in      = solver_param['injcetion_prim_state'][0]
    v_in        = solver_param['injcetion_prim_state'][1]
    P_in        = solver_param['injcetion_prim_state'][2]
    T_in        = solver_param['injcetion_prim_state'][3]
    Y_in        = solver_param['injcetion_prim_state'][4:]

    area_in     = solver_param['injector_face_area']

    mass_current= rho * vol
    mass_in     = rho_in * v_in * area_in * solver_param['dt']

    total_mass  = mass_current+mass_in

    # species mass conservation
    partial_mass_current = mass_current    * Y_ct[0,:,:].T
    partial_mass_inject  = mass_in * Y_in
    Y_mix                = (partial_mass_current + partial_mass_inject) / total_mass

    # moment conservation
    # vx_mix       = (rho_in * 0.01 + rho * u)/(rho_in + rho)


    # energy conservation
    gas_array_current.TPY = T[inj_indx], P[inj_indx], Y_ct[0,inj_indx]
    gas_array_inject.TPY  = T_in,P_in,Y_in.T

    int_energy_current    = np.squeeze(gas_array_current.int_energy_mass)
    int_energy_in         = np.squeeze(gas_array_inject.int_energy_mass)

    int_energy_mix        = (mass_current[inj_indx] * int_energy_current + mass_in * int_energy_in)/total_mass[inj_indx]
    vol_inj               = mass_in / gas_array_inject.density
    total_volume          = vol + vol_inj
    sp_volume_mix         = total_volume / total_mass[inj_indx]

    gas_array_mix.UVY     = int_energy_mix,sp_volume_mix,Y_mix[:,inj_indx].T

    internal_energy       = np.squeeze(gas_array_mix.int_energy_mass)

    internal_energy_tot   = internal_energy + (0.5 * (u[inj_indx] **2))

    # mass conservation 
    rho_mix = np.squeeze(gas_array_mix.density)
    Y_mix   = gas_array_mix.Y[0,:,:-1].T

    # momentum conservation using isentropic
    gamma_current                    = np.squeeze(gas_array_current.cp/gas_array_current.cv)
    speed_sound_current              = np.squeeze(gas_array_current.sound_speed)
    mach_current                     = u[inj_indx]/speed_sound_current
    P_tot_current                    = P[inj_indx] * (1+(gamma_current-1)/2*mach_current**2)**((gamma_current-1)/gamma_current)
    P_tot_mix                        = P_tot_current

    P_mix                            = np.squeeze(gas_array_mix.P)
    speed_sound_mix                  = np.squeeze(gas_array_mix.sound_speed)
    gamma_mix                        = np.squeeze(gas_array_mix.cp/gas_array_mix.cv)
    pressure_ratio                   = P_tot_mix/P_mix
    pressure_ratio[pressure_ratio<1] = 1

    mach_mix                         = np.sqrt(2/(gamma_current-1)*((pressure_ratio)**((gamma_mix-1)/gamma_mix)-1))


    u_mix                            = mach_mix * speed_sound_mix

    # replace the influenced cells states
    mass         = rho_mix * vol
    momx         = rho_mix * u_mix * vol
    energy       = (rho_mix * internal_energy_tot) * vol
    mass_species = rho_mix * Y_mix * vol

    Q_cons_inject= np.vstack((mass,momx,energy,mass_species))

    Q_cons_user[:,inj_indx] = Q_cons_inject

    smoothing_start_indx = (detonation[0]-10)%(solver_param['cell_number']+4)
    smoothing_end_indx   = (detonation[0]+10)%(solver_param['cell_number']+4)

    q_left  = Q_cons_user[:,smoothing_start_indx]
    q_right = Q_cons_user[:,smoothing_end_indx]

    x_knwon = np.array([0,20])
    y_known = np.array([q_left,q_right]).T

    f_interp = interp1d(x_knwon,y_known,kind='linear')

    x        = np.arange(0,20)

    q_ramp   = f_interp(x)

    if smoothing_start_indx < smoothing_end_indx:
        Q_cons_user[:, smoothing_start_indx:smoothing_end_indx] = q_ramp
    else:
        wrap_len = (solver_param['cell_number']+4) - smoothing_start_indx
        Q_cons_user[:, smoothing_start_indx:] = q_ramp[:, :wrap_len]
        Q_cons_user[:, :smoothing_end_indx]   = q_ramp[:, wrap_len:]

    state['Q_cons'] = Q_cons_user.ravel()

    return state

def source_calculator(solver_param,rom_param,state):

    # load the basic information
    Q_cons           = state['Q_cons']
    # state            = cons2prim_converter(solver_param,state)
    Q_prim           = state['Q_prim']
    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho              = Q_prim_user[0,:]
    vx               = Q_prim_user[1,:]
    press            = Q_prim_user[2,:]
    temp             = Q_prim_user[3,:]
    Y                = Q_prim_user[4:,:]

    Y_ct             = reshape_func.find_mass_fraction_full_cantera(Y)

    vol              = solver_param['vol']
    num_state_var    = solver_param['num_state_var']

    state['gas_array'].TPY = temp,press,Y_ct

    net_production_rate_ct = state['gas_array'].net_production_rates
    MW_species             = state['gas_array'].molecular_weights
    state['heat_release']  = np.squeeze(state['gas_array'].heat_release_rate)
    state['int_energy']    = np.squeeze(state['gas_array'].int_energy_mass)

    source_terms   = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    source_terms[3:,:]   = (net_production_rate_ct[0,:,:-1] * MW_species[:-1] * vol).T

    state['source_terms'] = source_terms.ravel()

    if solver_param['hyper'] == True:

        source_terms_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                               solver_param['num_state_var'],
                                                               state['source_terms'])

        state['source_terms'] = source_terms_int[rom_param['S_indx_solver']]

    state['d_flux_dx'] = state['d_flux_dx'] + state['source_terms']

    return state

def residual_calculator(solver_param,rom_param,state):
    
    if solver_param['flux_scheme'] == '1st Order Roe':

        state = first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state)

        if solver_param['viscous_flag']:

            state = first_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

    # apply flux vector
    state   = d_flux_dx_calculator(solver_param,rom_param,state)

    # apply source term 
    state = source_calculator(solver_param,rom_param,state)

    return state
