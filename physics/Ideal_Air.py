import numpy as np
import cantera as ct


from utils import reshape_func
from boundary_condition import bc_func
 
def prim2cons_converter(solver_param, state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']

    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    vol = solver_param['vol']

    gamma = 1.4

    rho = Q_prim_user[0,:]
    vx  = Q_prim_user[1,:] 
    P   = Q_prim_user[2,:]

    mass   = rho * vol
    momx   = rho * vx * vol
    energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

    state['Q_cons'] = np.vstack((mass,momx,energy)).ravel()

    return state

def cons2prim_converter(solver_param, state):

    vol         = solver_param['vol']
    gamma       = 1.4

    Q_cons      = state['Q_cons']
    Q_cons_user = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    mass        = Q_cons_user[0,:]
    momx        = Q_cons_user[1,:]
    energy      = Q_cons_user[2,:]
    
    rho = mass / vol
    vx  = momx / rho / vol
    P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)
    T   = P / rho / (ct.gas_constant / state['MW_gas']) 

    state['Q_prim'] = np.vstack((rho,vx,P,T)).ravel()

    return state

def second_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):
    
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

    
    gamma         = 1.4
    vol           = solver_param['vol']
    num_state_var = solver_param['num_state_var']

    en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)

    cell_num = len(rho)

    # total enthalpy
    htot = gamma/(gamma-1)*press/rho+0.5*vx**2

    # flux = np.zeros((num_state_var,cell_num+1))
    # diffusion = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        S_indx_user   = rom_param['S_indx_user']

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

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        left_state = Q_cons_user[:,range_flux_left]
        right_state= Q_cons_user[:,range_flux_right]

        Pinv =  np.zeros((2*len(S_indx_user),3,3))
        P    =  np.zeros((2*len(S_indx_user),3,3))
        lamb =  np.zeros((2*len(S_indx_user),3,3))

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        left_state = Q_cons_user[:,:-1]
        right_state= Q_cons_user[:,1:]

        Pinv =  np.zeros((cell_num-1,3,3))
        P    =  np.zeros((cell_num-1,3,3))
        lamb =  np.zeros((cell_num-1,3,3))

    R = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                  # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                    # {hat U}_{j+1/2}
    hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))   # {hat a}_{j+1/2}

    alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
    alph2=(gamma-1)/(amoy*amoy)

    Pinv[:,0,0] = 0.5*(alph1+umoy/amoy)
    Pinv[:,0,1] = -0.5*(alph2*umoy+1/amoy)
    Pinv[:,0,2] = alph2/2

    Pinv[:,1,0] = 1-alph1
    Pinv[:,1,1] = alph2*umoy
    Pinv[:,1,2] = -alph2

    Pinv[:,2,0] = 0.5*(alph1-umoy/amoy)
    Pinv[:,2,1] = -0.5*(alph2*umoy-1/amoy)
    Pinv[:,2,2] = alph2/2
    
    P[:,0,0] = 1
    P[:,0,1] = 1
    P[:,0,2] = 1

    P[:,1,0] = umoy-amoy
    P[:,1,1] = umoy
    P[:,1,2] = umoy+amoy

    P[:,2,0] = hmoy-amoy*umoy
    P[:,2,1] = 0.5*umoy*umoy
    P[:,2,2] = hmoy+amoy*umoy

    lamb[:,0,0] = abs(umoy-amoy)
    lamb[:,0,1] = 0
    lamb[:,0,2] = 0

    lamb[:,1,0] = 0
    lamb[:,1,1] = abs(umoy)
    lamb[:,1,2] = 0

    lamb[:,2,0] = 0
    lamb[:,2,1] = 0
    lamb[:,2,2] = abs(umoy+amoy)

    A = np.matmul(np.matmul(P,lamb),Pinv)

    dq = ((right_state-left_state).T)

    dq = dq[:,:,np.newaxis]

    diffusion = 0.5 * np.matmul(A,dq) / vol

    flux = np.zeros((num_state_var,cell_num+1))
    
    if solver_param['hyper'] == True:

        flux[0,range_flux_right] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                   - np.squeeze(diffusion[:,0,:])
        flux[1,range_flux_right] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)    - np.squeeze(diffusion[:,1,:])
        flux[2,range_flux_right] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))      - np.squeeze(diffusion[:,2,:])

    else:

        flux[0,1:-1] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                   - np.squeeze(diffusion[:,0,:])
        flux[1,1:-1] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)    - np.squeeze(diffusion[:,1,:])
        flux[2,1:-1] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))      - np.squeeze(diffusion[:,2,:])

    state['flux_cons'] = flux

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

def residual_calculator(solver_param,rom_param,state):
    
    if solver_param['flux_scheme'] == '2nd Order Roe':

        state = second_order_roe_inviscid_flux_calculator(solver_param,rom_param,state)

    # apply flux vector
    state   = d_flux_dx_calculator(solver_param,rom_param,state)

    return state
