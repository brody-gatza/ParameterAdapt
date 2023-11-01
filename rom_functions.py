import numpy as np
import time_integrator_functions

def precomputer(solver_param):

    training_data_path = solver_param['working_dir'] + "/FOM_cons.npy"
    POD_energy_limit   = 100-solver_param['pod_energy']
    training_data      = np.load(training_data_path)
    first_snapshot     = training_data[:,:,0]
    # first_snapshot     = np.mean(training_data,axis=2)
    centered_data      = training_data - first_snapshot[:,:,np.newaxis]

    l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
    norm_factor        = np.mean(l2_factors, axis=1)

    cen_norm_data      = centered_data / norm_factor[:, np.newaxis, np.newaxis]

    tall_thin_data = np.zeros((cen_norm_data.shape[0] * cen_norm_data.shape[1], cen_norm_data.shape[2]))
    
    for indx in range(cen_norm_data.shape[2]):

        tall_thin_data[:, indx] = cen_norm_data[:, :, indx].reshape(-1)

    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

    square_sum_singular_values = np.sum(S**2)

    POD_res_energy = [(1 - np.sum(S[:indx]**2) / square_sum_singular_values) * 100 for indx in range(len(S))]

    truncation_indx = np.where(np.array(POD_res_energy) < POD_energy_limit )

    print(f'to capture '+ str(solver_param['pod_energy']) +' percent of energy '+ str(truncation_indx[0][0])+ ' mode must be used')

    norm_matrix_size = S.size

    norm_matrix = np.zeros((norm_matrix_size, norm_matrix_size))

    segment_size = norm_matrix_size // len(norm_factor)

    for i in range(norm_matrix_size):
        value_index = i // segment_size
        norm_matrix[i, i] = norm_factor[value_index]

    basis         = V[:,0:truncation_indx[0][0]]
    # basis         = V
    denormalizor  = norm_matrix
    normalizor    = np.linalg.inv(denormalizor)
    q_ref         = first_snapshot.ravel(order='C')

    return basis , normalizor, denormalizor , q_ref


def order_reducer(solver_param,d_mass_dt,d_momx_dt,d_energy_dt,basis , normalizor, denormalizor , q_ref,full_cons_results):

    RES = np.vstack((d_mass_dt[2:-2],d_momx_dt[2:-2],d_energy_dt[2:-2]))
    RES = RES.ravel(order='C')

    dt       = solver_param['dt']

    if solver_param['rom_method'] == 'Galerkin':

        Q0_red           = basis.T @ full_cons_results[:,2:-2].ravel(order='C')
        dQ_red_dt        = basis.T @ normalizor @ RES
        Q_red            = time_integrator_functions.explicit_fd_euler(Q0_red , dQ_red_dt   , dt)
        Q_full_order     = q_ref + (denormalizor @ basis @ Q_red) 

    desired_shape        = full_cons_results[:,2:-2].shape
    Q                    = Q_full_order.reshape(( desired_shape[0] , desired_shape[1]))

    mass   = np.hstack(( Q[0,0], Q[0,0] , Q[0,:] , Q[0,-1] ,Q[0,-1] ))
    momx   = np.hstack(( Q[1,0], Q[1,0] , Q[1,:] , Q[1,-1] ,Q[1,-1] ))
    energy = np.hstack(( Q[2,0], Q[2,0] , Q[2,:] , Q[2,-1] ,Q[2,-1] ))

    return mass, momx, energy



    