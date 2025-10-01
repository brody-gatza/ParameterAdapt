import numpy as np
import os

def quad_kron(X):

    N, w = X.shape
    X2_list = []

    for j in range(w):
        prod = X[:, j:j+1] * X[:, j:]  # Broadcasting to multiply column j with columns j to w
        X2_list.append(prod)

    kron_prod = np.hstack(X2_list)

    return kron_prod

def u_func(t,param,solver_param):

    inputs             = param['inputs']
    t0                 = param['rom_train_end_iter'] * solver_param['dt']
    tf                 = solver_param['num_step']    * solver_param['dt']
    num_remaining_iter = solver_param['num_step'] - param['rom_train_end_iter']
    t_eval             = np.linspace(t0, tf, num_remaining_iter)
    i                  = np.argmin(np.abs(t_eval - t))

    return inputs[:, i]

# def u_func(t,param,solver_param):

#     density_bc = solver_param['bc_data'][0,0]

#     parts = density_bc.split('/')

#     value = float(parts[0])  # value
#     A     = float(parts[1])  # amplitude
#     f     = float(parts[2])  # frequency

#     t = state['time']

#     Q_prim_user[4:,0:2] = value + A*np.sin(f*t)

#     return inputs[:, i]

def rom_rhs(t, q, c, A, H, B, param,solver_param,u_func):

    q = q.reshape(-1, 1)  # Ensure column shape

    kron_prod = np.outer(q,q)
    kron_prod = kron_prod[np.triu_indices(len(q), k=0)]

    quad_term = H @ kron_prod # (r x r^2) × (r^2 x 1)
    u = u_func(t,param,solver_param).reshape(-1, 1)   # Get control at time t
    # u = 0   # Get control at time t
    # rhs = c + A @ q + quad_term + B @ u
    rhs = c.reshape(-1,1) + A @ q + quad_term.reshape(-1,1) + B @ u
    # rhs = c.reshape(-1,1) + A @ q + quad_term.reshape(-1,1)

    return rhs.flatten()

def shpae_data_finder(param):

    # find some basics about shape of the function
    sample_cons_file = np.load(os.path.join(param['result_dir'],'cons_prim','0iteration_cons.npy'))

    param['num_state_var'] = np.shape(sample_cons_file)[0]
    param['num_cell']      = np.shape(sample_cons_file)[1]
    param['num_snapshot']  = len(np.arange(param['rom_train_start_iter'],
                                param['rom_train_end_iter'],
                                param['rom_train_step_iter']))

    return param

def assemble_data(param):

    # read basic params
    num_state_var = param['num_state_var']
    num_cell      = param['num_cell']     
    num_snapshot  = param['num_snapshot']

    cons_train_data = np.zeros((num_state_var,num_cell,num_snapshot))
    res_train_data  = np.zeros((num_state_var,num_cell,num_snapshot))

    iter_list = np.arange(param['rom_train_start_iter'],
                        param['rom_train_end_iter'],
                        param['rom_train_step_iter'])

    # iterate and assemble data
    for indx,iter in enumerate(iter_list):
        
        cons_train_data[:,:,indx]=np.load(os.path.join(param['result_dir'],'cons_prim',str(iter)+'iteration_cons.npy'))
        res_train_data[:,:,indx] =np.load(os.path.join(param['result_dir'],'res',str(iter)+'iteration_res.npy'))
    
    param['cons_train_data'] = cons_train_data
    param['res_train_data']  = res_train_data

    return param

def init_rom_param(param,rom_param):

    param['basis']                = rom_param['basis']
    param['q_ref']                = rom_param['q_ref']
      
    param['norm']                 = rom_param['normalizor']  
    param['denorm']               = rom_param['denormalizor']
    param['cent_norm_train_data'] = rom_param['SROM_training_win'][:,:-1]
    param['qr_0']                 = param['basis'].T @ param['cent_norm_train_data'][:,-1]

    return param


def create_snapshots_matrix(param):

    # non intrusive data snapshots
    param['ni_snapshots'] = param['basis'].T@param['cent_norm_train_data']

    return param

def create_time_derivative_matrix(param):

    # non intrusive time derivative snapshots
    q_ref          = param['cons_train_data'][:,:,0]
    res_train_data = param['res_train_data'] - q_ref[:,:,np.newaxis]
    res_train_data = res_train_data.reshape(-1, param['num_snapshot'])
    res_train_data = res_train_data * param['norm'][:,np.newaxis]

    param['ni_time_derivative'] = param['basis'].T@res_train_data

    return param

def create_inputs_matrix(param):

    num_input    = 2*param['num_state_var']
    num_cell     = param['num_cell']
    num_snapshot = param['num_snapshot']

    inputs = np.zeros((num_input,num_snapshot))
    
    inputs[0,:] = param['cent_norm_train_data'][(0*num_cell),:]
    inputs[1,:] = param['cent_norm_train_data'][(1*num_cell),:]
    inputs[2,:] = param['cent_norm_train_data'][(2*num_cell),:]

    inputs[3,:] = param['cent_norm_train_data'][(1*num_cell)-1,:]
    inputs[4,:] = param['cent_norm_train_data'][(2*num_cell)-1,:]
    inputs[5,:] = param['cent_norm_train_data'][(3*num_cell)-1,:]

    param['inputs'] = inputs

    return param

def create_known_data_matrix(param):

    num_mode     = np.shape(param['basis'])[1]
    num_snapshot = param['num_snapshot']

    num_column = int(1 + num_mode + (num_mode*(num_mode+1)/2))

    identity_column = np.ones((num_snapshot,1))
    # input_column    = np.zeros((num_mode,num_snapshot))

    kron_prod = quad_kron(param['ni_snapshots'].T)

    # outer_prod = np.outer(param['ni_snapshots'], param['ni_snapshots'])
    # kron_prod  = np.triu(outer_prod,k=0)

    # kron_prod = np.kron(param['ni_snapshots'], param['ni_snapshots'])

    identity_column = np.ones((num_snapshot,1))

    inputs_column   = param['inputs']

    knwon_data = np.hstack((identity_column,
                            param['ni_snapshots'].T,
                            kron_prod,
                            inputs_column.T))

    param['known_data'] = knwon_data

    return param

def solve_optimization_problem(param):

    knonw_data_shape = np.shape(param['known_data'])[1]

    reg_fact_values = param['reg_fact_value']

    reg_fact_matrix = np.eye(knonw_data_shape)*reg_fact_values

    # coefficient matrix
    A = (param['known_data'].T @ param['known_data'] + 
         reg_fact_matrix.T @ reg_fact_matrix) 
    
    # RHS vector
    b = param['known_data'].T @ param['ni_time_derivative'].T

    # solution vector (Ax=b)
    X = np.linalg.solve(A,b)

    param['ni_operator'] = X.T

    return param