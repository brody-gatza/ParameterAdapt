import os
import numpy as np
import matplotlib.pyplot as plt

# cum_sum    = np.load(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\cumsum.npy'    )
# moving_avg = np.load(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\moving_avg.npy')
# sub_angle  = np.load(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\sub_angle.npy' )


# plt.figure()
# plt.plot(sub_angle,color='tab:blue')
# plt.plot(moving_avg,color='tab:red')
# plt.plot(cum_sum,color='tab:red')

# plt.show()


fig, ax = plt.subplots(1,1)

line1, = plt.plot([],[],lw=4,c='black')
line2, = plt.plot([],[],lw=2,c='tab:red' ,ls='-')
line3, = plt.plot([],[],lw=2,c='tab:blue',ls='--')

x = np.linspace(0,1,500)

ax.set_ylim([0.8,2])
ax.set_xlim([-0.05,1.05])

ax.set_xlabel('x [m]')
ax.set_ylabel('Density [kg/m3]')

adaptive_error = np.zeros(49)
self_adaptive_error = np.zeros(49)

iter_list = np.arange(1000,100000,1000)

adaptive_error = np.zeros(len(iter_list))
self_adaptive_error = np.zeros(len(iter_list))

for indx, iter in enumerate(iter_list):

    hrom = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\Hybrid ROM_results\cons_prim',str(iter)+'iteration_prim.npy'))
    arom = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\Adaptive ROM_results\cons_prim',str(iter)+'iteration_prim.npy'))
    fom  = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\FOM_Results',str(iter)+'iteration_prim.npy'))

    line1.set_data(x,fom[0,:])
    line2.set_data(x,arom[0,:])
    line3.set_data(x,hrom[0,:])

    ax.legend(['FOM','Adaptive ROM','Self-Adaptive ROM'])

    # if iter >= 14000:
        
    #     line2.set_color('tab:red')

    adaptive_error[indx]      = np.mean(np.linalg.vector_norm(fom-arom,axis=1)/np.linalg.vector_norm(fom,axis=1))
    self_adaptive_error[indx] = np.mean(np.linalg.vector_norm(fom-hrom,axis=1)/np.linalg.vector_norm(fom,axis=1))


    plt.pause(0.1)

plt.show()


fig, ax = plt.subplots(1,1)

t = np.linspace(0,0.01,len(iter_list))

ax.plot(t,adaptive_error,color='black',ls='-',lw=1.5,label='Adaptive ROM')
ax.plot(t,self_adaptive_error,color='red',ls='--',lw=1.5,label='Self Adaptive ROM')

ax.set_xlabel('Time')
ax.set_ylabel('Variable Averaged Normalized Error')
ax.set_yscale('log')
ax.legend()

ax.set_ylim([1e-5,1e0])

plt.show()