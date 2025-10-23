import os
import numpy as np

data_dir    = r'C:\GIT_Fork\ROMify\examples\1D_RDE\FOM_results_11cycles\cons_gathered.npy'
pod_energy  = 95
cell_number = 200

main_data = np.load(data_dir)

training_data_cons = main_data[:,:,5000:10000:10]

# number of snapshot
num_snapshot = len(training_data_cons[0,0,:])

# reference profile
# q_ref = training_data_cons[:,:,0]
q_ref = np.mean(training_data_cons,axis=2)

# center data 
centered_data = training_data_cons - q_ref[:,:,np.newaxis]

# normalizing factors
l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
norm_factor        = np.mean(l2_factors, axis=1)

# centered_normalized data
cen_norm_data = centered_data / norm_factor[:, np.newaxis, np.newaxis]

# data matrix
tall_thin_data = cen_norm_data.reshape(-1, num_snapshot)

# perform SVD
V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

# POD residual energy check
square_sum_singular_values = np.sum(S**2)
cumulative_energy          = np.cumsum(S**2)
POD_res_energy             = (1 - (cumulative_energy / square_sum_singular_values)) * 100

POD_energy_limit = 100-pod_energy

truncation_indx = np.where(np.array(POD_res_energy) < POD_energy_limit)[0][0]

# finalize the basis
basis = V[:,0:truncation_indx]

# wrap up and exit the function
denormalizor = np.repeat(norm_factor, cell_number)
normalizor   = 1/denormalizor

basis               = basis
q_ref               = q_ref.ravel()
norm                = normalizor
denorm              = denormalizor

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
line0, = ax.plot([],[],c='tab:blue',ls='-')
line1, = ax.plot([],[],c='tab:red' ,ls='--')
x = np.arange(0,200)

iter_list = np.arange(5000,10000,1)

for indx , iter in enumerate(iter_list):

    data = main_data[:,:,iter]

    q_cons = data.ravel()
    q_cons_cent = q_cons - q_ref
    q_cons_norm = q_cons_cent * norm

    qr = basis.T @ q_cons_norm

    q_cons_rep = q_ref + denorm *(basis@qr)

    line0.set_data(x,q_cons[400:600])
    line1.set_data(x,q_cons_rep[400:600])

    ax.set_ylim([-50,450])
    ax.set_xlim([-1,201])

    plt.pause(0.001)

print('Done!')
