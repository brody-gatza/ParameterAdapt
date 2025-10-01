import numpy as np
import scipy as sc

from utils import reshape_func

def hyper_precompute(solver_param,rom_param,static_basis=True,):

    num_cell     = solver_param['cell_number']
    num_samples  = int(solver_param['sampling_rate'] * num_cell / 100)
    num_state_var= solver_param['num_state_var']
    basis        = rom_param['basis']

    # find the required sample first

    if solver_param['sampling_method'] == 'GNAT':

        S_indx_user = GNAT_sample_point_finder(basis,num_cell)

    elif solver_param['sampling_method'] == 'QDEIM':

        S_indx_user = QDEIM_sample_point_finder(basis,num_cell)

    elif solver_param['sampling_method'] == 'QDEIM+R':

        S_indx_user = QDEIM_R_sample_point_finder(basis,num_samples,num_cell)

    elif solver_param['sampling_method'] == 'GappyPODE':

        S_indx_user = GappyPODE_sample_point_finder(basis,num_samples,num_cell)

    elif solver_param['sampling_method'] == 'ECSW':

        S_indx_user = ECSW_sample_point_finder(solver_param,rom_param)

    elif solver_param['sampling_method'] == 'FGS':

        ref_state   = rom_param['Q_bar']

        S_indx_user = FGS_sample_point_finder(basis,num_samples,num_cell,ref_state,num_state_var)
    
    S_indx_solver = reshape_func.user2solver_indx_converter(S_indx_user,
                                                            num_state_var,
                                                            num_cell)
    
    S_indx_user   = np.sort(S_indx_user)
    S_indx_solver = np.sort(S_indx_solver)

    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver

    if static_basis:

        # compute the sampling precomputed term
        hyper_precompute = basis @ np.linalg.pinv(basis[S_indx_solver,:])
        rom_param['hyper_precompute'] = hyper_precompute

    return rom_param

def GNAT_sample_point_finder(basis,num_cell):

    num_state_var = int(np.shape(basis)[0]/num_cell)
    num_modes     = np.shape(basis)[1]

    max_mode_indx  = np.argsort(np.abs(basis[:,0]),axis=0)[-2:]

    # pick the first sample
    S_indx_solver  = max_mode_indx

    for mode in range(2,num_modes):

        u_l = basis[:,mode].reshape(-1,1)

        c   = np.linalg.pinv(basis[S_indx_solver,0:mode])@u_l[S_indx_solver]
        
        r   = u_l - (basis[:,0:mode] @ c)

        new_sample = np.argmax(np.abs(r))

        S_indx_solver = np.append(S_indx_solver,new_sample)

    S_indx_user = reshape_func.solver2user_indx_converter(S_indx_solver,num_cell)

    S_indx_user = np.unique(S_indx_user)

    return S_indx_user
    
def QDEIM_sample_point_finder(basis,num_cell):

    n, m = basis.shape

    num_selected_samples = 0
    counter = 0

    Q, R, P = sc.linalg.qr(basis.T, mode='full',pivoting=True)

    S_indx_solver = P[:m]

    S_indx_user   = reshape_func.solver2user_indx_converter(S_indx_solver,num_cell)

    S_indx_user   = np.unique(S_indx_user)

    num_selected_samples = len(S_indx_user)
    
    while num_selected_samples < m:
        
        start_indx = m + counter
        end_indx   = m + counter + 1

        new_sample = P[start_indx:end_indx]

        S_indx_solver = np.append(S_indx_solver,new_sample)

        S_indx_user   = reshape_func.solver2user_indx_converter(S_indx_solver,num_cell)

        S_indx_user   = np.unique(S_indx_user)

        num_selected_samples = len(S_indx_user)

        counter = counter + 1

    S_indx_user = np.sort(S_indx_user)

    S_indx_user = S_indx_user.astype(int)

    return S_indx_user

def QDEIM_R_sample_point_finder(basis,num_samples,num_cell):

    S_indx_user = QDEIM_sample_point_finder(basis,num_cell)

    range_elements = np.arange(0,num_cell)
    options2choose = np.delete(range_elements,S_indx_user)

    n_sample_rand = num_samples - len(S_indx_user)

    if n_sample_rand<=0:

        return S_indx_user

    random_selected_samples          = np.random.choice(options2choose,size = int(n_sample_rand))

    S_indx_user = np.append(S_indx_user,random_selected_samples)
    
    S_indx_user = np.sort(S_indx_user)

    return S_indx_user

def GappyPODE_sample_point_finder(basis,num_samples,num_cell):

    _, _, p = sc.linalg.qr(basis.T, mode='full', pivoting=True)

    p = p[:basis.shape[1]]

    for i in range(len(p) + 1, num_samples + 1):

        _, S, W = np.linalg.svd(basis[p, :], full_matrices=False)

        g = S[-2]**2 - S[-1]**2

        Ub = np.dot(W.T, basis.T)

        r = g + np.sum(Ub**2, axis=0)

        r = r - np.sqrt((g + np.sum(Ub**2, axis=0))**2 - 4 * g * Ub[-1, :]**2)

        I = np.argsort(r)[::-1]

        e = 0

        while I[e] in p:

            e += 1

        p = np.append(p, I[e])


    S_indx_solver = p

    S_indx_user   = reshape_func.solver2user_indx_converter(S_indx_solver,num_cell)

    return S_indx_user

def ECSW_sample_point_finder(solver_param,rom_param):

    tall_thin_data = rom_param['cent_norm_train_data']

    dq_dt          = tall_thin_data

    W               = np.zeros((tall_thin_data.shape[0],1))

    num_state_var = solver_param['num_state_var']
    state_var_indx= np.arange(num_state_var)

    S_indx_user   = np.array([0])
    S_indx_solver = S_indx_user + (state_var_indx * solver_param['cell_number'])

    num_selected_samples = 1

    while True:

        sampled_dq_dt  = dq_dt[S_indx_solver,:]

        W              = dq_dt @ np.linalg.pinv(sampled_dq_dt)

        error_vec      = dq_dt - W @ sampled_dq_dt

        error_norm     = np.linalg.norm(error_vec)/np.linalg.norm(dq_dt)

        if error_norm < 1e-6:

            return S_indx_user

        else:

            # update counter
            num_selected_samples   = num_selected_samples + 1

            # where largest error occurs
            new_sample_indx_solver = np.max(np.abs(error_vec))
            row_index, col_index   = np.unravel_index(np.argmax(error_vec), error_vec.shape)

            # add new sample in mesh scale
            new_sample_indx_user   = row_index%solver_param['cell_number']
            S_indx_user            = np.append(S_indx_user,new_sample_indx_user)

            # make sure the new sample does not exist
            S_indx_user            = np.fromiter(dict.fromkeys(S_indx_user), dtype=S_indx_user.dtype)
            S_indx_user            = S_indx_user[0:num_selected_samples]

            # update samples in solver scale
            S_indx_solver = np.add.outer(S_indx_user ,(state_var_indx * solver_param['cell_number'])).ravel()

def FGS_sample_point_finder(basis,num_samples,num_cell,ref_state,num_state_var):

    S_indx_user = QDEIM_sample_point_finder(basis,num_cell)

    range_elements = np.arange(0,num_cell)
    options2choose = np.delete(range_elements,S_indx_user)

    n_sample_fgs = num_samples - len(S_indx_user)

    if n_sample_fgs<=0:

        return S_indx_user

    Q_future            = ref_state
    # Q_future_int_solver = reshape_func.solver_eliminate_ghost(num_cell,num_state_var,Q_future)
    Q_future_solver_user= reshape_func.results_solver2user_converter(num_state_var,num_cell,Q_future)
    Q_future_int_user   = Q_future_solver_user[:,2:-2]
    
    ### future shock ###
    rho_bar_shock = Q_future_int_user[2,:]

    first_derv_density  = np.gradient(rho_bar_shock)

    high_grad_area_indx = np.argsort(first_derv_density)
    high_grad_area_indx = np.delete(high_grad_area_indx,S_indx_user)

    fgs_selected_samples = high_grad_area_indx[0:n_sample_fgs]

    S_indx_user = np.append(S_indx_user,fgs_selected_samples)
    
    S_indx_user = np.sort(S_indx_user)

    return S_indx_user