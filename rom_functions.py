import numpy as np
import time_integrator_functions

def precomputer(solver_param):

    training_data_path_cons = solver_param['working_dir'] + "/FOM_cons.npy"
    training_data_path_prim = solver_param['working_dir'] + "/FOM_prim.npy"
    POD_energy_limit   = 100-solver_param['pod_energy']
    training_data_cons = np.load(training_data_path_cons)
    training_data_prim = np.load(training_data_path_prim)
    first_snapshot     = training_data_cons[:,:,0]
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

    norm_matrix_size = state_var_num * cell_num

    norm_matrix_diag = np.array([])

    for indx in range(0,state_var_num):

        temp = np.full(cell_num , norm_factor[indx])
        norm_matrix_diag = np.append(norm_matrix_diag,temp)

    norm_matrix = np.diag(norm_matrix_diag)

    # basis         = V[:,0:truncation_indx[0][0]]
    # basis         = V[:,0:500]
    basis         = V
    normalizor    = norm_matrix
    denormalizor  = np.linalg.inv(normalizor)
    q_ref         = first_snapshot.ravel(order='C')

    return basis , normalizor, denormalizor , q_ref , training_data_prim

def hyper_precomputer(basis,S_indx_solver):

    pcc = basis @ np.linalg.pinv(basis[S_indx_solver,:])

    return pcc

def order_reducer(solver_param,d_mass_dt,d_momx_dt,d_energy_dt,basis , normalizor, denormalizor , q_ref,q_red,pcc):

    if solver_param['calc_rom'] and solver_param['hyper']: 

        RES = np.vstack((d_mass_dt,d_momx_dt,d_energy_dt))
        RES = RES.ravel(order='C')
        RES = pcc @ RES

    elif solver_param['calc_rom']: 

        RES = np.vstack((d_mass_dt[2:-2],d_momx_dt[2:-2],d_energy_dt[2:-2]))
        RES = RES.ravel(order='C')


    dt       = solver_param['dt']

    if solver_param['rom_method'] == 'Galerkin':

        Q0_red           = q_red
        dQ_red_dt        = basis.T @ normalizor @ RES
        Q_red            = time_integrator_functions.explicit_fd_euler(Q0_red , dQ_red_dt   , dt)
        Q_full_order     = q_ref + (denormalizor @ basis @ Q_red) 

    desired_shape        = [3,solver_param['cell_number']]
    Q                    = Q_full_order.reshape(( desired_shape[0] , desired_shape[1]))

    mass   = np.hstack(( Q[0,0], Q[0,0] , Q[0,:] , Q[0,-1] ,Q[0,-1] ))
    momx   = np.hstack(( Q[1,0], Q[1,0] , Q[1,:] , Q[1,-1] ,Q[1,-1] ))
    energy = np.hstack(( Q[2,0], Q[2,0] , Q[2,:] , Q[2,-1] ,Q[2,-1] ))

    return mass, momx, energy , Q_red


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

    S_indx_user = np.unique(S_indx_user)
    S_indx_user = np.sort(S_indx_user)
    S_indx_user = S_indx_user.astype(int)

    return S_indx_user

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

    