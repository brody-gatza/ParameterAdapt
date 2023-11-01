
import numpy as np
from matplotlib import pyplot as plt

data_cons = np.load(r"C:\GIT_Fork\ROMify\examples\standing_flame\results\FOM_cons.npy")
data_prim = np.load(r"C:\GIT_Fork\ROMify\examples\standing_flame\results\FOM_prim.npy")
# IC_prim   = np.load(r"C:\GIT_Fork\ROMify\examples\standing_flame\rom_basis\prim_IC.npy")
IC_cons   = data_cons[:,:,3000]
IC_prim   = data_prim[:,:,3000]

data_cons = data_cons[:,:,3000:5000]
dim       = np.shape(data_cons)

centered_data =  data_cons - IC_cons[:,:,np.newaxis]

norm_factor = np.linalg.norm(centered_data, axis=2)

cen_norm_data = centered_data / norm_factor[:,:,np.newaxis]

tall_thin_data = cen_norm_data.reshape(dim[0]*dim[1],dim[2])

V, S, UT = np.linalg.svd(tall_thin_data , full_matrices=False)

norm_factor = norm_factor.flatten()

P_shape = norm_factor.shape[0]

zero_matrix = np.zeros((P_shape , P_shape))

np.fill_diagonal(zero_matrix,norm_factor)

# np.save(r"C:\GIT_Fork\ROMify\examples\standing_flame\rom_basis\cons_basis.npy",V)
# np.save(r"C:\GIT_Fork\ROMify\examples\standing_flame\rom_basis\cons_norm_factor.npy",zero_matrix)
# np.save(r"C:\GIT_Fork\ROMify\examples\standing_flame\rom_basis\cons_IC.npy",IC_cons)
# np.save(r"C:\GIT_Fork\ROMify\examples\standing_flame\rom_basis\prim_IC.npy",IC_prim)

fig , ax = plt.subplots(2,2)

ax[0,0].plot(IC_cons[0,:])
ax[0,1].plot(IC_cons[1,:])
ax[1,0].plot(IC_cons[2,:])

plt.show()

energy = np.zeros(np.shape(S))

for indx in range(1,len(S)):
    
    energy[indx] = (np.sum(S[0:indx])**2)/(np.sum(S)**2) * 100

num_mode = np.where(energy >= 99.999)[0][0]
print(num_mode)
