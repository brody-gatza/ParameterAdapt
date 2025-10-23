import os
import numpy as np

def results_solver2user_converter(num_state_var,num_cell,Q):

    Q_reshaped = np.reshape(Q,(num_state_var,num_cell+4))

    return Q_reshaped

def results_user2solver_converter(Q):

    Q_reshaped = np.ravel(Q)

    return Q_reshaped

def user2solver_indx_converter(S_indx_user,num_consv_var,num_cell):

    state_var_indx= np.arange(num_consv_var)

    S_indx_solver = np.add.outer(S_indx_user ,(state_var_indx * num_cell)).ravel()

    return S_indx_solver

def solver2user_indx_converter(S_indx_solver,num_cell):

    S_indx_user = S_indx_solver%num_cell

    S_indx_user = S_indx_user.astype(int)

    return S_indx_user

def solver_add_ghost(cell_number,num_var,Q_int):

    Q_int_user   = results_solver2user_converter(num_var,cell_number-4,Q_int)
    Q_int_full   = np.column_stack((Q_int_user[:,0], Q_int_user[:,0], Q_int_user , Q_int_user[:,-1] , Q_int_user[:,-1]))
    Q_full_solver= results_user2solver_converter(Q_int_full)
    
    return Q_full_solver

def solver_eliminate_ghost(cell_number,num_var,Q_full):

    Q_full_user = results_solver2user_converter(num_var,cell_number,Q_full)
    Q_int_user = Q_full_user[:,2:-2]
    Q_int_solver = results_user2solver_converter(Q_int_user)

    return Q_int_solver

def find_mass_fraction_full(MF):
    MF_last_row = 1.0 - np.sum(MF, axis=0)
    MF_full = np.vstack((MF, MF_last_row))
    MF_full[MF_full == 0] = 1e-30
    return MF_full

def find_mass_fraction_full_cantera(MF):

    MF = find_mass_fraction_full(MF)
    MF_reshaped = MF.T[np.newaxis, :, :]  # shape: (1, N_points, N_species)

    return MF_reshaped

def assemble_snapshots(solver_param):
        
    print('Assembling Snapshots!')

    list_snapshots     = np.arange(solver_param['training_start_iter'],
                                   solver_param['training_end_iter'  ],
                                   solver_param['training_step_iter' ])
    
    num_snapshot = len(list_snapshots)

    training_data_cons = np.zeros((solver_param['num_state_var'],solver_param['cell_number'],num_snapshot))

    # bring all of snapshots into one matrix
    for indx , iter in enumerate(list_snapshots):
        
        file_name_cons = str(iter)+'iteration'+'_cons.npy'
        training_data_cons[:, :, indx] = np.load(os.path.join(solver_param['training_data_dir'], file_name_cons))

    print('Assembling Snapshots Completed!')
    return training_data_cons