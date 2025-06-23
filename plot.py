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

line1, = plt.plot([],[],c='black')
line2, = plt.plot([],[],c='tab:red' ,ls='--')
line3, = plt.plot([],[],c='tab:blue',ls='--')

x = np.linspace(0,1,500)

ax.set_ylim([200,400])
ax.set_xlim([0,1])

ax.set_xlabel('x [m]')
ax.set_ylabel('Density [kg/m3]')

for iter in range(1000,50000,1000):

    hrom = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\Hybrid ROM_results\cons_prim',str(iter)+'iteration_prim.npy'))
    arom = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\Adaptive ROM_results\cons_prim',str(iter)+'iteration_prim.npy'))
    fom  = np.load(os.path.join(r'C:\GIT_Fork\ROMify\examples\supersonic_flow\FOM_Results',str(iter)+'iteration_prim.npy'))

    line1.set_data(x,fom[-1,:])
    line2.set_data(x,arom[-1,:])
    line3.set_data(x,hrom[-1,:])

    # if iter >= 14000:
        
    #     line2.set_color('tab:red')



    plt.pause(1)

plt.show()
