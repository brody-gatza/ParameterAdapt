import sys
import os
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np


iter_50k = np.load(r"C:\Users\mohag\Desktop\FOM_Results\50000iteration_prim.npy")
temp_ic  = iter_50k[:-1,:]
np.save(r"C:\Users\mohag\Desktop\after_reflection_ic.npy",temp_ic)


dir_results_fom = r"C:\Users\mohag\Desktop\FOM_results"
# dir_results_arom = r"C:\Users\mohag\Desktop\Adaptive ROM_results\cons_prim"
# dir_results_arom_basis = r"C:\Users\mohag\Desktop\Adaptive ROM_results\basis"
# dir_results_arom_sample = r"C:\Users\mohag\Desktop\Adaptive ROM_results\samples_solver"
max_iter    = 51000
freq_vis    = 1000


sample_cons_snapshot = np.load(os.path.join(dir_results_fom, '0iteration_cons.npy'))
sample_cons_shape    = np.shape(sample_cons_snapshot)

sample_prim_snapshot = np.load(os.path.join(dir_results_fom, '0iteration_prim.npy'))
sample_prim_shape    = np.shape(sample_prim_snapshot)

# sample_basis_snapshot = np.load(r"C:\Users\mohag\Desktop\Adaptive ROM_results\basis\10iteration_basis.npy")
# sample_basis_shape    = np.shape(sample_basis_snapshot)

# sample_samples_solver_snapshot = np.load(r"C:\Users\mohag\Desktop\Adaptive ROM_results\samples_solver\10iteration_samples_solver.npy")
# sample_samples_solver_shape    = np.shape(sample_samples_solver_snapshot)

data_cons_fom = np.zeros((sample_cons_shape[0],sample_cons_shape[1],int(max_iter/freq_vis)))
data_prim_fom = np.zeros((sample_prim_shape[0],sample_prim_shape[1],int(max_iter/freq_vis)))

# data_cons_arom = np.zeros((sample_cons_shape[0],sample_cons_shape[1],int(max_iter/freq_vis)))
# data_prim_arom = np.zeros((sample_prim_shape[0],sample_prim_shape[1],int(max_iter/freq_vis)))

# basis_matrix   = np.zeros((sample_basis_shape[0],sample_basis_shape[1],int(max_iter/freq_vis)))

# sample_matrix   = np.zeros((sample_samples_solver_shape[0],int(max_iter/freq_vis)),dtype=int)

print('Assembling Snapshots!')

counter = 0

# bring all of snapshots into one matrix
for i in range(0,max_iter,freq_vis):

    file_name_cons = str(i)+'iteration'+'_cons.npy'
    file_name_prim = str(i)+'iteration'+'_prim.npy'

    data_cons_fom[:, :, counter] = np.load(os.path.join(dir_results_fom, file_name_cons))
    data_prim_fom[:, :, counter] = np.load(os.path.join(dir_results_fom, file_name_prim))

    # data_cons_arom[:, :, counter] = np.load(os.path.join(dir_results_arom, file_name_cons))
    # data_prim_arom[:, :, counter] = np.load(os.path.join(dir_results_arom, file_name_prim))

    counter=counter+1


# counter = 0

# # bring all of snapshots into one matrix
# for i in range(1000,max_iter,freq_vis):

#     file_name_basis = str(i)+'iteration'+'_basis.npy'

#     basis_matrix[:, :, counter] = np.load(os.path.join(dir_results_arom_basis,file_name_basis))

#     counter=counter+1

# counter = 0
# # bring all of snapshots into one matrix
# for i in range(1000,max_iter,freq_vis):

#     file_name_sample = str(i)+'iteration'+'_samples_solver.npy'

#     sample_matrix[:, counter] = np.load(os.path.join(dir_results_arom_sample,file_name_sample))

#     counter=counter+1


# # calculate proj error

# proj_error = np.zeros((int(max_iter/freq_vis)))

# for i in range(0,int(max_iter/freq_vis)):

#     I = np.eye(sample_basis_shape[0],sample_basis_shape[0])

#     proj_matrix = I - (basis_matrix[:,:,i]@basis_matrix[:,:,i].T)

#     proj_error[i] = np.linalg.norm(proj_matrix)


# counter=0

# sample_error = np.zeros((int(max_iter/freq_vis)))

# for i in range(0,int(max_iter/freq_vis)):

#     pinv_ST_V = np.linalg.pinv(basis_matrix[sample_matrix[:, counter],:,i])

#     sample_error[i] = np.linalg.norm(pinv_ST_V)


# plt.figure()
# plt.plot(proj_error)
# # plt.yscale('log')

# plt.figure()
# plt.plot(sample_error)
# # plt.yscale('log')

fig , ax = plt.subplots(2,2)

x=np.linspace(0,0.12,500)

for iter in range(0,int(max_iter/freq_vis)):


    ax[0,0].cla()
    ax[1,0].cla()
    ax[0,1].cla()
    ax[1,1].cla()


    hr1     = data_prim_fom[0,: ,iter].ravel()
    u1      = data_prim_fom[1 ,: ,iter].ravel()
    P1      = data_prim_fom[2 ,: ,iter].ravel()
    T1      = data_prim_fom[3 ,: ,iter].ravel()

    # rho2    = data_prim_arom[0,: ,iter].ravel()
    # u2      = data_prim_arom[1 ,: ,iter].ravel()
    # P2      = data_prim_arom[2 ,: ,iter].ravel()
    # T2      = data_prim_arom[3 ,: ,iter].ravel()

    ax[0,0].plot(x,hr1        ,color='tab:blue'  , label='FOM')
    ax[1,0].plot(x,u1         ,color='tab:green' , label='FOM')
    ax[0,1].plot(x,P1         ,color='tab:orange', label='FOM')
    ax[1,1].plot(x,T1         ,color='tab:red'   , label='FOM')

    # ax[0,0].plot(x,rho2      ,linestyle='--',color='tab:blue'  ,label='AROM')
    # ax[1,0].plot(x,u2        ,linestyle='--',color='tab:green' ,label='AROM')
    # ax[0,1].plot(x,P2        ,linestyle='--',color='tab:orange',label='AROM')
    # ax[1,1].plot(x,T2        ,linestyle='--',color='tab:red'   ,label='AROM')

    # ax[0,0].scatter(x[s_indx] , rho[s_indx]      ,color='black')
    # ax[1,0].scatter(x[s_indx] , u[s_indx]        ,color='black')
    # ax[0,1].scatter(x[s_indx] , a_P[s_indx]      ,color='black')
    # ax[1,1].scatter(x[s_indx] , a_T[s_indx]      ,color='black')

    # ax[0,0].legend()
    # ax[1,0].legend()
    # ax[0,1].legend()
    # ax[1,1].legend()

    #     ax.set_xlabel('x[m]')
    # ax[0,0].set_xlabel('x[m]')
    # ax[1,0].set_xlabel('x[m]')
    # ax[0,1].set_xlabel('x[m]')
    # ax[1,1].set_xlabel('x[m]')

    # ax[0,0].set_ylabel('Heat Release[W/m3]')
    # ax[1,0].set_ylabel('Velocity[m/s]')
    # ax[0,1].set_ylabel('Pressure[Pa]')
    # ax[1,1].set_ylabel('Temperature[K]')
    #     ax.set_ylabel('P')

    # ax[0,1].set_ylim(95e3,107e3)
    # ax[1,0].set_ylim(-10,10)

    #     counter = counter + 1
    #     counter2 = counter2 + 100

    print(iter*freq_vis)

    plt.pause(1e-6)



plt.show()





