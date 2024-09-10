import sys
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt


# res_hist        = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\FOM 3500 snapshots Explicit - FD Euler res_hist.npy")
# basis           = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis_save.npy")
# S_indx_solver   = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\S_indx_solver_save_all.npy")


# res_fom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_fom.npy")
# res_rom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_rom.npy")


fom = np.load(r"C:\GIT_Fork\ROMify\examples\transient_flame\FOM 9999 snapshots Explicit - FD Euler prim.npy")

# basis = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis.npy")
# norm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\norm.npy")
# denorm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\denorm.npy")
# qref = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\q_ref.npy")

T = fom[3,:,6000].ravel()
P = fom[2,:,6000].ravel()
u = fom[1,:,6000].ravel()

slice_source = fom[:,201,6000]

fom[:,0:201,6000] = slice_source[:,np.newaxis]

np.save(r"C:\GIT_Fork\ROMify\examples\transient_flame\steady_IC_trans.npy",fom[:,:,6000])

T = fom[3,:,6000].ravel()
P = fom[2,:,6000].ravel()
u = fom[1,:,6000].ravel()

fig , ax = plt.subplots(3,1)

ax[0].plot(T,label='T')
ax[1].plot(P,label='P')
ax[2].plot(u,label='u')
# ax.plot(q_replica,linestyle='--',label='ROM no HR')
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()

# for iter in range(0,10000,50):

#     ax[0].cla()
#     ax[1].cla()
#     ax[2].cla()

#     T = fom[3,:,iter].ravel()
#     P = fom[2,:,iter].ravel()
#     u = fom[1,:,iter].ravel()

#     # q_replica = qref+(denorm*(basis@(basis.T@(norm*(q-qref)))))



#     print(iter)

# basis = basis[:,0:11]

# for iter in range(0,3500,100):

#     ax.cla()

#     f = res_hist[:,iter]

#     f = np.reshape(f,(12,1004))

#     f = f[:,2:-2].ravel()

#     f_replica = basis @ np.linalg.pinv(basis[S_indx_solver,:]) @ f[S_indx_solver]

#     ax.plot(f,label='FOM res')
#     ax.plot(f_replica,linestyle='--',label='Hyper-Reduced res')

#     ax.legend()

#     plt.pause(1e-6)



#     print(iter)



