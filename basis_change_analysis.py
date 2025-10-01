import numpy as np
import scipy.linalg as sp_al
import matplotlib.pyplot as plt
import os 
from multiprocessing import Pool
from matplotlib.animation import FFMpegFileWriter
from tqdm import tqdm
import cantera as ct

# compute dv
def compute_dv(args):

    iter,dir = args

    basis_n   = np.load(os.path.join(dir,'basis',str(iter)+'iteration_basis.npy'))
    basis_nm1 = np.load(os.path.join(dir,'basis',str(iter-1000)+'iteration_basis.npy'))
    dv        = np.linalg.norm(basis_n - basis_nm1) / np.linalg.norm(basis_nm1)

    # print(iter)

    return dv 

def subspace_angles_calc(args):

    iter,dir  = args        

    basis_n   = np.load(os.path.join(dir,'basis',str(iter)+'iteration_basis.npy'))
    basis_nm1 = np.load(os.path.join(dir,'basis',str(iter-1000)+'iteration_basis.npy'))

    sub_angle = np.linalg.norm(
            np.rad2deg(
            sp_al.subspace_angles(basis_n[:,:],basis_nm1[:,:])
            )
            )

    return sub_angle

def cons2prim_combustion(gas_array,cons):

    vol         = 0.01/500

    Q_cons      = cons
    Q_cons_user = np.reshape(cons,(12,500))

    mass        = Q_cons_user[0,:]
    momx        = Q_cons_user[1,:]
    energy      = Q_cons_user[2,:]
    mass_species= Q_cons_user[3:,:]
    
    rho = mass / vol
    vx  = momx / rho / vol
    MF  = mass_species / rho / vol

    MF_shape = np.shape(MF)

    # add the 1 last species
    MF_last_row = np.zeros(MF_shape[1])

    for indx in range(0,MF_shape[1]):

        MF_last_row[indx] = 1.0 - np.sum(MF[:,indx])

    MF = np.vstack((MF,MF_last_row))

    MF[MF==0] = 1e-30

    MF_shape = np.shape(MF)

    # reshape the MF array suitable for cantera

    MF_reshaped = np.zeros( (1,MF_shape[1],MF_shape[0]) )

    for indx in range(0,MF_shape[0]):

        MF_reshaped[:,:,indx] = MF[indx,:]

    MF_ct = MF_reshaped

    sp_vol = 1/rho

    internal_energy = (energy/vol/rho)-(0.5*vx**2)

    gas_array.UVY = internal_energy,sp_vol,MF_ct

    T = np.squeeze(gas_array.T)
    P = np.squeeze(gas_array.P)

    prim = np.vstack((rho,vx,P,T,MF)).ravel()

    prim = np.reshape(prim , (14,500))

    return prim


if __name__ == '__main__':

    # mode = 'write'
    mode = 'read'

    dir = r"C:\Users\mohag\OneDrive\Desktop\1D_RDE_Basis_Results"
    # dir = r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\Adaptive ROM_results"
    # dir = r"C:\GIT_Fork\ROMify\examples\supersonic_flow\FOM_results"
    # dir = r"C:\GIT_Fork\ROMify\examples\supersonic_flow\Hybrid ROM_results"

    if mode =='write':

        iterations = range(2000,4770000,1000)

        args_list = [(iter, dir) for iter in iterations]

        with Pool(processes=24) as pool:

            results = list( tqdm(pool.imap(subspace_angles_calc,args_list) ,total = len(args_list) ))

        dBasis = np.array(results)

        np.save(os.path.join(dir,'dBasis_normalized.npy'),dBasis)

    elif mode=='read':

        "plot first two modes"

        # fig, ax = plt.subplots(1,2,figsize=(15,5))

        # x   = np.linspace(0,0.1,500)
        # DOF = np.arange(0,500)

        # line0,   = ax[0].plot([],[],c='tab:blue')

        # line10 , = ax[1].plot([],[],ls='-',c='tab:blue' ,label='first mode')
        # line11 , = ax[1].plot([],[],ls='--',c='tab:red',label='second mode')

        # ax[0].set_xlabel('x')
        # ax[0].set_ylabel('Pressure [Pa]')

        # ax[1].set_xlabel('DOF')
        # ax[1].set_xlabel('DOF')
        # ax[1].legend()

        # write = FFMpegFileWriter(fps=30,bitrate=8000)

        # with write.saving(fig,
        #                   os.path.join(dir,"variation of basis inc svd.mp4"),
        #                   dpi=500):

        #     for iter in tqdm(range(11,43960,50)):

        #         P      = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_prim.npy'))[2,:]
        #         basis  = np.load(os.path.join(dir,'basis',str(iter)+'iteration_basis_inc_svd.npy'))
                
        #         line0.set_data(x,P)

        #         line10.set_data(DOF,basis[0:500,0])
        #         line11.set_data(DOF,basis[0:500,1])


        #         ax[0].set_xlim([0,0.1])
        #         ax[0].set_ylim([99000,103525])

        #         ax[1].set_xlim([0,500])
        #         ax[1].set_ylim([-0.1,0.1])

        
        #         # plt.pause(1e-3)

        #         # print(iter)

        #         write.grab_frame()

        "plot dV"

        # fig, ax = plt.subplots(1,1)

        # # dBasis = np.load(os.path.join(dir,'dBasis.npy'))
        # dBasis = np.load(os.path.join(dir,'dBasis_normalized.npy'))

        # ax.plot(dBasis[::50],c='tab:blue')

        # identical_iter = [int(n*6280/50) for n in range(0,7)]
        # # ax.scatter(identical_iter,dBasis[::6280],c='black')

        # # ax.set_ylabel('$|dV|^2_2$')
        # ax.set_ylabel(r'$\frac{|\mathbf{V}^{n} - \mathbf{V}^{n-1}|_2^2}{|\mathbf{V}^{n-1}|_2^2}$')
        # ax.set_xlabel('Iteration')

        # plt.show()


        "plot first two modes at perturb. freq."

        # fig, ax = plt.subplots(1,1)

        # identical_iter = [n*6280 for n in range(1,7)]

        # color_list = ['tab:red','tab:blue','tab:green','black','yellow','tab:orange']

        # n = 1

        # for iter in identical_iter:

        #     basis  = np.load(os.path.join(dir,'basis',str(iter)+'iteration_basis.npy'))

        #     ax.plot(basis[:,0],ls='-' ,c = color_list[n-1], label=str(n) + 'Cycle 1st Mode')
        #     # ax.plot(basis[:,1],ls='--',c = color_list[n-1], label=str(n) + 'Cycle 2nd Mode')

        #     n = n + 1

        # ax.legend()
        # ax.set_ylabel('-')
        # ax.set_xlabel('DOF')

        # plt.show()

        "dv at pert. freq."

        # identical_iter = [n*6280 for n in range(1,7)]

        # n = 1

        # dV = np.zeros(len(identical_iter))

        # for iter in range(1,len(identical_iter)):

        #     name_ex   = os.path.join(dir,'basis',str(identical_iter[iter-1])+'iteration_basis.npy')
        #     name_next = os.path.join(dir,'basis',str(identical_iter[iter])+'iteration_basis.npy')
            
        #     basis_ex    = np.load(name_ex)
        #     basis_next  = np.load(name_next)

        #     dV[iter] = np.linalg.norm(basis_next-basis_ex)/np.linalg.norm(basis_ex)

        #     n = n + 1
        
        # identical_iter = np.array(identical_iter)

        # ax.plot(identical_iter[1:]/50,dV[1:],c='black',marker='o')

        # plt.show()

        "incremental SVD"

        # def incremental_svd_column(U, S, Vt, a_new_col):
        #     """
        #     Incremental SVD when adding a new column to A.
        #     A ≈ U @ np.diag(S) @ Vt, with A ∈ ℝ^{m × n}

        #     Parameters:
        #     - U, S, Vt: SVD of current matrix A
        #     - a_new_col: new column to add (shape: (m,))
        #     - eps: numerical threshold for rank detection

        #     Returns:
        #     - U_new, S_new, Vt_new: updated SVD of [A | a_new_col]
        #     """
        #     m = U.shape[0]
        #     k = len(S)

        #     # Project new column onto current U basis
        #     p = U.T @ a_new_col  # shape: (k,)
        #     r = a_new_col - U @ p  # residual
        #     r_norm = np.linalg.norm(r)

        #     if r_norm < 1e-10:
        #         # The new column is in the span of U
        #         q = np.linalg.pinv(np.diag(S)) @ p
        #         Vt_new = np.hstack([Vt, q[:, np.newaxis]])
        #         return U, S, Vt_new

        #     # Orthonormalize residual
        #     u_new = r / r_norm

        #     # Expand U
        #     U_new = np.hstack([U, u_new[:, np.newaxis]])

        #     # Augmented K matrix for small SVD
        #     K = np.zeros((k + 1, k + 1))
        #     K[:k, :k] = np.diag(S)
        #     K[:k, -1] = p
        #     K[-1, -1] = r_norm

        #     # Compute SVD of small (k+1)x(k+1) matrix
        #     U_bar, S_new, Vt_bar = np.linalg.svd(K)

        #     # Update U and Vt
        #     U_new = U_new @ U_bar
        #     Vt_new = np.zeros((k + 1, Vt.shape[1] + 1))
        #     Vt_new[:k, :Vt.shape[1]] = Vt
        #     Vt_new[-1, :] = 0  # placeholder row
        #     Vt_new = Vt_bar.T @ Vt_new
        #     Vt_new[:, -1] = Vt_bar[-1, :]  # fill last column

        #     return U_new, S_new, Vt_new

        # qref   = np.load(os.path.join(dir,'q_ref',str(10)+'iteration_q_ref.npy'))
        # norm   = np.load(os.path.join(dir,'norm',str(10)+'iteration_norm.npy'))
        # denorm = np.load(os.path.join(dir,'denorm',str(10)+'iteration_denorm.npy'))

        # init_window = np.zeros((len(qref),10))

        # for iter in range(10):

        #     cons = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_cons.npy')).ravel()

        #     init_window[:,iter] = (cons-qref)*norm

        # U, S, Vt = np.linalg.svd(init_window, full_matrices=False)

        # for iter in tqdm(range(11,43960)):

        #     new_snapshot = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_cons.npy')).ravel()
        #     new_snapshot_cen_norm = (new_snapshot - qref)*norm

        #     U , S , Vt = incremental_svd_column(U, S, Vt, new_snapshot_cen_norm)

        #     np.save(os.path.join(dir,'basis',str(iter)+'iteration_basis_inc_svd.npy'),U)

    "Plot At Freq Iter"

    # fig, ax = plt.subplots(1,1)

    # cycle = 0

    # iterations = [0,6283,12566,18850,25133,31416,37699,43982,50265,56549,62832,69115,75398,81681,87965,94248]

    # for iter in iterations:

    # #     # P = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_prim.npy'))[2,:]
    #     # rho = np.load(os.path.join(dir,str(iter)+'iteration_prim.npy'))[0,:]
    #     rho = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_prim.npy'))[0,:]

    # #     # ax.plot(P , label=str('Cycle = ' + str(cycle)))
    #     ax.plot(rho , label=str('Cycle = ' + str(cycle)))

    #     cycle += 1

    # ax.legend()
    # plt.show()

    "Subspace Angle"

    # iterations = iterations = range(2000,4770000,1000)

    # args_list = [(iter, dir) for iter in iterations]

    # with Pool(processes=16) as pool:

    #     results = list( tqdm(pool.imap(subspace_angles_calc,args_list) ,total = len(args_list) ))

    # subspace_angels = np.array(results)

    # np.save(os.path.join(dir,'subspace_angles_deg_first_five_modes_only_norm.npy'),subspace_angels)

    sub_angle = np.load(os.path.join(dir,'subspace_angles_deg_first_two_modes_only_norm.npy'))
    sub_angle[250:] = sub_angle[250:] - 80
    sub_angle = np.abs(sub_angle)

    ######################################## CUMSUM ########################################
    mean = np.mean(sub_angle)

    cusum = np.cumsum(sub_angle-mean)

    cusum = (cusum - np.min(cusum)) / (np.max(cusum) - np.min(cusum))

    ######################################## Moving Average ########################################

    window_size = 10

    moving_avg = np.convolve(sub_angle, np.ones(window_size)/window_size, mode='same')

    # indx = np.where((moving_avg<=1) & (cusum <= 0.01))[0][0]

    # indx = int(2e4)

    # print(indx)

    ######################################## Dynamic Threshold ########################################

    # dyc_moving_avg = np.zeros_like(sub_angle)+1e32
    # dyc_cumsum     = np.zeros_like(sub_angle)+1e32

    # for iter in range(1,len(sub_angle),1):

    #     # compute cumsum on the fly

    #     mean_d = np.mean(sub_angle[0:iter])

    #     cusum_d = np.cumsum(sub_angle[0:iter]-mean_d)

    #     dyc_cumsum[0:iter] = (cusum_d - np.min(cusum_d)) / (np.max(cusum_d) - np.min(cusum_d))

    #     # compute moving average on the fly

    #     dyc_moving_avg[0:iter] = np.convolve(sub_angle[0:iter], np.ones(window_size)/window_size, mode='same')

    #     if np.any(dyc_cumsum<=0.01) and np.any(dyc_moving_avg<=0.05):

    #         print('The basis good enough! Time to Take Off!')

    #         indx = np.where((dyc_cumsum<=0.01) & (dyc_moving_avg<=0.05))[0][0]

    #         print('index: ' + str(indx) + ' is found as the transition points')

    #         break

    
    ######################################## Plot ########################################

    fig, ax = plt.subplots(1,1)

    axx = ax.twinx()

    x = np.linspace(0,0.15,len(sub_angle))

    line0, = ax.plot(x,sub_angle ,color='tab:blue'    , label='Subspace Angles')
    line1, = ax.plot(x,moving_avg,color='black' , ls='-'  ,label='Moving Average (w=10)')
    line2, =axx.plot(x,cusum,color='tab:red',ls='--', label='CUSUM')
    # ax.scatter(x[change_points],sub_angle[change_points]*0+45)

    # identical_iter = np.array([int(n*6280) for n in range(0,8)])
    # ax.vlines(identical_iter,0,90,color='grey',linestyles='--',label='Cycles',lw=0.5)

    ax.set_ylabel('Subspace Angles')
    axx.set_ylabel('Cum. Sum Values')
    ax.set_xlabel('Time [ms]')

    ax.set_xlim([0,0.15])

    # Merge legends
    lines = [line0, line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right')

    plt.show()

    ######################################## Dynamic Plot ########################################

    # basis  = np.load(os.path.join(dir,'basis',str(indx)+'iteration_basis.npy'))
    # qref   = np.load(os.path.join(dir,'q_ref',str(10)+'iteration_q_ref.npy'))
    # norm   = np.load(os.path.join(dir,'norm',str(10)+'iteration_norm.npy'))
    # denorm = np.load(os.path.join(dir,'denorm',str(10)+'iteration_denorm.npy'))


    # gas = ct.Solution('h2o2.yaml')
    # gas.basis = 'mass'
    # gas_array = ct.SolutionArray(gas,(1,int(500)))

    # fig, ax = plt.subplots(1,1,figsize=(12,8))

    # x   = np.linspace(0,0.01,500)
    # DOF = np.arange(0,500)

    # line0 , = ax.plot([],[],ls='-' ,c='black' ,label='FOM',lw=3)
    # line1 , = ax.plot([],[],ls='--',c='tab:blue' ,label='A-ROM')
    # line2 , = ax.plot([],[],ls='--',c='tab:red',label='SA-ROM')

    # ax.set_ylabel('Pressure [Pa]')
    # ax.set_xlabel('x [-]')
    # ax.legend()

    # proj_error = np.zeros(43960/50)

    # write = FFMpegFileWriter(fps=30,bitrate=5000)

    # with write.saving(fig,
    #                     os.path.join(dir,"automated_transition.mp4"),
    #                     dpi=150):

    #     for iter in range(11,43960,50):

    #         cons     = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_cons.npy'))
    #         prim     = np.load(os.path.join(dir,'cons_prim',str(iter)+'iteration_prim.npy'))
    #         energ    = cons[2,:]
    #         P_arom   = prim[2,:]
    #         q_arom   = cons.ravel()

    #         cent_norm_q = norm*(q_arom - qref)

    #         q_rep = qref+denorm*(basis@basis.T@cent_norm_q)

    #         q_rep = np.reshape(q_rep,(12,500))
    #         # proj_error[iter] = np.linalg.norm(q_rep-q_arom) / np.linalg.norm(q_arom)

    #         # line0.set_data(x,P)

    #         prim_rep = cons2prim_combustion(gas_array,q_rep)

    #         P_rep = prim_rep[2,:]

    #         line0.set_data(x,P_arom)
    #         line1.set_data(x,P_arom)

    #         if iter < indx:

    #             line2.set_data(x,P_arom*0)

    #         else:

    #             line2.set_data(x,P_rep)


    #         ax.set_xlim([0,0.01])
    #         ax.set_ylim([99000,103525])

    #         # ax[1].autolim(axis='both')
    #         # ax[1].relim()
    #         # ax[1].autoscale(axis='both')

    #         # ax[1].set_xlim([0,0.01])
    #         # ax[1].set_ylim([-3,3])


    #         # plt.pause(1e-3)

    #         print(iter)

    #         write.grab_frame()

    "Subspace Angle at Period Iter"

    # identical_iter = [n*6280 for n in range(1,7)]

    # n = 1

    # subspace_angle = np.zeros((2,len(identical_iter)))

    # for iter in range(1,len(identical_iter)):

    #     name_ex   = os.path.join(dir,'basis',str(identical_iter[iter-1])+'iteration_basis.npy')
    #     name_next = os.path.join(dir,'basis',str(identical_iter[iter])+'iteration_basis.npy')
        
    #     basis_ex    = np.load(name_ex)
    #     basis_next  = np.load(name_next)

    #     subspace_angle[:,iter] = np.rad2deg(sp_al.subspace_angles(basis_ex[:,0:2],basis_next[:,0:2]))

    #     n = n + 1
    
    # identical_iter = np.array(identical_iter)

    # fig, ax= plt.subplots(1,1)

    # ax.plot(identical_iter,subspace_angle[0,:],marker='o')
    # ax.plot(identical_iter,subspace_angle[1,:],marker='o')

    # plt.show()

    "Dot Products"











    
