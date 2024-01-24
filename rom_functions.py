import time
import numpy as np
import scipy as sc
import solver_functions
import time_integrator_functions
from matplotlib import pyplot as plt

def precomputer(solver_param,state):

    print('Initializing the ROM mode!')

    if solver_param['solver_mode'] == 'ROM':

        training_data_path_cons = solver_param['FOM_result_dir']
        training_data_path_prim = training_data_path_cons.replace(' cons.npy',' prim.npy')

        training_data_cons      = np.load(training_data_path_cons)
        training_data_prim      = np.load(training_data_path_prim)
        first_snapshot          = training_data_cons[:,:,0]        

    elif solver_param['solver_mode'] == 'Adaptive ROM':

        iter = solver_param['iter']

        training_data_cons = state['cons_results_save']
        training_data_prim = state['prim_results_save']
        first_snapshot     = training_data_cons[:,:,iter-1]


    POD_energy_limit   = 100-solver_param['pod_energy']
    # first_snapshot     = np.mean(training_data,axis=2)

    state_var_num      = len(training_data_cons[:,0,0])
    snapshot_num       = len(training_data_cons[0,0,:])
    cell_num           = solver_param['cell_number']

    centered_data      = np.empty((state_var_num,cell_num,snapshot_num))

    for indx in range(0,snapshot_num):

        centered_data[:,:,indx] = training_data_cons[:,:,indx] - first_snapshot

    l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
    norm_factor        = np.mean(l2_factors, axis=1)

    cen_norm_data      = np.zeros((state_var_num,cell_num,snapshot_num))

    for i in range(0,state_var_num):

        for j in range(0,snapshot_num):

            cen_norm_data[i,:,j] = centered_data[i,:,j] / norm_factor[i]


    tall_thin_data = np.zeros((state_var_num * cell_num, snapshot_num))
    
    for indx in range(0,snapshot_num):

        tall_thin_data[:, indx] = cen_norm_data[:,:,indx].ravel(order='C')

    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=True)

    square_sum_singular_values = np.sum(S**2)

    POD_res_energy = [(1 - np.sum(S[:indx]**2) / square_sum_singular_values) * 100 for indx in range(len(S))]

    truncation_indx = np.where(np.array(POD_res_energy) < POD_energy_limit )

    print(f'to capture '+ str(solver_param['pod_energy']) +' percent of energy '+ str(truncation_indx[0][0])+ ' mode must be used')

    time.sleep(1.5)

    norm_matrix_size = state_var_num * cell_num

    norm_matrix_diag = np.array([])

    for indx in range(0,state_var_num):

        temp = np.full(cell_num , norm_factor[indx])
        norm_matrix_diag = np.append(norm_matrix_diag,temp)

    norm_matrix = np.diag(norm_matrix_diag)

    # basis         = V[:,0:truncation_indx[0][0]]
    basis         = V[:,0:30]
    # basis         = V
    denormalizor    = norm_matrix
    normalizor      = np.linalg.inv(denormalizor)
    q_ref           = first_snapshot.ravel(order='C')

    print(str(len(basis[0,:])) + ' modes are going to be used in this simulation')

    time.sleep(1.5)

    rom_param     = {}

    rom_param['basis']              = basis
    rom_param['normalizor']         = normalizor
    rom_param['denormalizor']       = denormalizor
    rom_param['q_ref']              = q_ref
    rom_param['training_data_prim'] = training_data_prim

    Q_cons_solver = state['Q_cons']
    Q_cons_user   = solver_functions.results_solver2user_converter(cell_num,Q_cons_solver)
    Q_cons_interior_user = Q_cons_user[:,2:-2]
    Q_cons_interior_solver = solver_functions.results_user2solver_converter(Q_cons_interior_user) 
    rom_param['q_red0'] = basis.T @ Q_cons_interior_solver

    num_consv_var = 3
    
    S_indx_user   = np.arange(0,solver_param['cell_number'])
    pcc           = 0
    S_indx_solver = user2solver_indx_converter(S_indx_user,num_consv_var,solver_param['cell_number'])

    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver
    rom_param['hyper_precompute'] = pcc

    return rom_param

def hyper_precomputer(basis,S_indx_solver):

    pcc = basis @ np.linalg.pinv(basis[S_indx_solver,:])

    return pcc

def red2full_state_calculator(solver_param,rom_param,state):

    if solver_param['rom_method'] == 'Galerkin':

        state = solver_functions.residual_calculator(solver_param,rom_param,state)

        if solver_param['hyper'] == False:

            RES_solver       = state['d_flux_dx']
            RES_user         = solver_functions.results_solver2user_converter(solver_param['cell_number'],RES_solver)
            RES              = RES_user[:,2:-2].ravel()

        else:
            pcc              = rom_param['hyper_precompute']
            RES_solver       = state['d_flux_dx']   
            RES              = pcc @ RES_solver


        Q0_red           = rom_param['q_red0']
        q_ref            = rom_param['q_ref']
        normalizor       = rom_param['normalizor']
        denormalizor     = rom_param['denormalizor']
        basis            = rom_param['basis']

        dQ_red_dt        = basis.T @ normalizor @ RES

        state['Q_cons']  = Q0_red

        state['d_flux_dx']= dQ_red_dt

        state            = time_integrator_functions.advance_time(solver_param,state)

        Q_red            = state['Q_cons']

        Q_full_order_solver= q_ref + (denormalizor @ basis @ Q_red) 

        Q_full_order_user  = solver_functions.results_solver2user_converter(solver_param['cell_number']-4,Q_full_order_solver)

        Q_full_order_user  = np.column_stack((Q_full_order_user[:,0], Q_full_order_user[:,0], Q_full_order_user , Q_full_order_user[:,-1] , Q_full_order_user[:,-1]))

        state['Q_cons']    = solver_functions.results_user2solver_converter(Q_full_order_user)

        rom_param['q_red0']    = Q_red

        state['d_flux_dx'] = RES_solver


    return state , rom_param

def user2solver_indx_converter(S_indx_user,num_consv_var,num_cell):

    num_selected_cell = np.size(S_indx_user)

    S_indx_solver     = np.zeros((num_consv_var*num_selected_cell),dtype=int)
    
    for j in np.arange(0,num_consv_var):
        
        start = j*num_selected_cell
        end   = (j+1)*num_selected_cell

        S_indx_solver[start:end] = S_indx_user + j * num_cell

    return S_indx_solver

def solver2user_indx_converter(S_indx_solver,num_cell):

    all_vars_vector_size = np.size(S_indx_solver)
    
    S_indx_user = np.array([])

    for j in range(0,all_vars_vector_size):

        if S_indx_solver[j] >= num_cell:

            coeff = np.floor(S_indx_solver[j]/num_cell)

            new_indx = S_indx_solver[j] - coeff * num_cell

        else: 

            new_indx = S_indx_solver[j]

        S_indx_user = np.append(S_indx_user, new_indx)  

    return S_indx_user

def sample_point_finder(solver_param,rom_param):

    basis = rom_param['basis']

    if solver_param['sampling_method'] == 'DEIM':

            S_indx_user = DEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'QDEIM':
            
            S_indx_user = QDEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'Gappy POD + E':
            
            num_samples = 150

            S_indx_user = GappyPODE_sample_point_finder(basis,num_samples,solver_param['cell_number'])
    
    num_consv_var  = 3

    S_indx_solver = user2solver_indx_converter(S_indx_user,num_consv_var,solver_param['cell_number'])

    pcc       = hyper_precomputer(basis,S_indx_solver)

    rom_param['hyper_precompute'] = pcc
    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver

    return rom_param

def DEIM_sample_point_finder(basis,num_cell):

    max_mode_value = np.max(np.abs(basis[:,0]))
    max_mode_indx  = np.argmax(np.abs(basis[:,0]),axis=0)

    S_indx_temp = max_mode_indx

    num_grids = np.shape(basis)[0]
    num_modes = np.shape(basis)[1]
    
    

    U = basis[:,0]
    U = U.reshape(-1, 1)
    P = np.zeros((num_grids,1))
    P[max_mode_indx] = 1


    for mode in range(1,num_modes):
        
        u_l = basis[:,mode].reshape(-1,1)
        c = np.linalg.solve(np.transpose(P) @ U , np.transpose(P) @ u_l)
        r = u_l - U @ c
        new_max_value = np.max(np.abs(r))
        new_indx      = np.argmax(np.abs(r))
        new_P_vector  = np.zeros((num_grids,1))
        new_P_vector[new_indx] = 1 
        U = np.hstack([U,u_l])
        P = np.hstack([P,new_P_vector])
        S_indx_temp = np.hstack([S_indx_temp,new_indx])

    S_indx_solver = S_indx_temp

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    return S_indx_user

def QDEIM_sample_point_finder(U,num_cell):

    n, m = U.shape

    num_selected_samples = 0
    counter = 0

    Q, R, P = sc.linalg.qr(U.T, mode='full',pivoting=True)

    S_indx_solver = P[:m]

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    S_indx_user   = np.unique(S_indx_user)

    num_selected_samples = len(S_indx_user)
    
    while num_selected_samples < m:
        
        start_indx = m + counter
        end_indx   = m + counter + 1

        new_sample = P[start_indx:end_indx]

        S_indx_solver = np.append(S_indx_solver,new_sample)

        S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

        S_indx_user   = np.unique(S_indx_user)

        num_selected_samples = len(S_indx_user)

        counter = counter + 1

    S_indx_user = np.sort(S_indx_user)

    S_indx_user = S_indx_user.astype(int)

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

    breakpoint()

    S_indx_solver = p

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    return S_indx_user

def adaptive_rom_progress(solver_param,rom_param,state,iter):

    if iter <= int(solver_param['init_training_win']):

        if iter != int(solver_param['init_training_win']):
            
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

        elif iter == int(solver_param['init_training_win']):
            
            # Q 101
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

            # V 101 and Qr 101
            rom_param = precomputer(solver_param,state)

    else:

            solver_param['hyper'] = False

            Q_tilda_old = state['Q_cons']
            Q_tilda_old_user = solver_functions.results_solver2user_converter(solver_param['cell_number'],Q_tilda_old)
            Q_tilda_old_user_int = Q_tilda_old_user[:,2:-2]
            Q_tilda_old_solver_int = solver_functions.results_user2solver_converter(Q_tilda_old_user_int)

            state , rom_param = red2full_state_calculator(solver_param,rom_param,state)

            Q_red_new = rom_param['q_red0']

            Q_tilda_new = state['Q_cons']
            Q_tilda_new_user = solver_functions.results_solver2user_converter(solver_param['cell_number'],Q_tilda_new)
            Q_tilda_new_user_int = Q_tilda_new_user[:,2:-2]
            Q_tilda_new_solver_int = solver_functions.results_user2solver_converter(Q_tilda_new_user_int)

            state['Q_cons'] = Q_tilda_old

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)


            Q_bar_new = state['Q_cons']
            Q_bar_new_user = solver_functions.results_solver2user_converter(solver_param['cell_number'],Q_bar_new)
            Q_bar_new_user_int = Q_bar_new_user[:,2:-2]
            Q_bar_new_solver_int = solver_functions.results_user2solver_converter(Q_bar_new_user_int)


            del_basis = ( ((Q_bar_new_solver_int-Q_tilda_new_solver_int).reshape(-1,1))@(Q_red_new.reshape(1,-1)) ) / (np.linalg.norm(Q_red_new) ** 2)
            rom_param['basis']  = rom_param['basis'] + del_basis

            new_basis  = rom_param['basis']
            q_ref      = rom_param['q_ref']
            denormalizor = rom_param['denormalizor']

            Q_bar_new_int_check = q_ref + (denormalizor @ new_basis @ Q_red_new) 

            diff = Q_bar_new_int_check - Q_bar_new_solver_int

            breakpoint()


            state['Q_cons'] = Q_tilda_new

    
    return state, solver_param , rom_param



            






            












    # update sampling points

    # rom_param = sample_point_finder(solver_param,rom_param)

    # S_indx_solver           = rom_param['S_indx_solver']
    # S_indx_user             = rom_param['S_indx_user']
    # num_samples             = len(S_indx_user)

    # interpolation_error      = full_state_combo - (rom_param['basis'] @ np.linalg.pinv(rom_param['basis'][S_indx_solver]) @ full_state_combo[S_indx_solver])
    # interpolation_error_indx = np.argsort(np.abs(interpolation_error))[::-1]
    # S_indx_solver            = np.sort(interpolation_error_indx[0:num_samples])
    # S_indx_user              = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])

    # check to avoid repeating indicies

    # counter = 0

    # n , m = rom_param['basis'].shape

    # S_indx_user   = np.unique(S_indx_user[0:m])

    # num_selected_samples = len(S_indx_user)

    # while num_selected_samples < m:
    
    #     start_indx = num_samples + counter
    #     end_indx   = num_samples + counter + 1

    #     new_sample = interpolation_error_indx[start_indx:end_indx]

    #     S_indx_solver = np.append(S_indx_solver,new_sample)

    #     S_indx_user   = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])

    #     S_indx_user   = np.unique(S_indx_user)

    #     num_selected_samples = len(S_indx_user)

    #     counter = counter + 1

    # # S_indx_user = np.sort(S_indx_user)
    
    # S_indx_user = np.arange(100,400)

    # S_indx_solver = user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])

    # S_indx_user = S_indx_user.astype(int)

    # pcc         = hyper_precomputer(rom_param['basis'],S_indx_solver)

    # rom_param['hyper_precompute'] = pcc
    # rom_param['S_indx_solver'] = S_indx_solver
    # rom_param['S_indx_user']   = S_indx_user


         






        

    