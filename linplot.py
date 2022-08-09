import numpy as np
import matplotlib.pyplot as plt
numRuns = 11

fig, axs = plt.subplots(2,3)

for i in range(numRuns):
    namestr = 'run'+str(i+1)+'.csv'
    array = np.genfromtxt(namestr,delimiter=',',skip_header=True)
    axs[0,0].plot(array[:,0],array[:,1],'.')
    axs[0,1].plot(array[:,0],array[:,2],'.')
    axs[0,2].plot(array[:,0],array[:,3],'.')
    axs[1,0].plot(array[:,0],array[:,4],'.')
    axs[1,1].plot(array[:,0],array[:,5],'.')
    axs[1,2].plot(array[:,0],array[:,6],'.')


for ax in np.reshape(axs,(6)):
    ax.set_xlabel('e_dd')
axs[0,0].set_ylabel('obliquity eta')
axs[0,1].set_ylabel('transverse width / (QHO width)')
axs[0,2].set_ylabel('longitudinal width / (QHO width)')
axs[1,0].set_ylabel('contrast parameter')
axs[1,1].set_ylabel('period / (QHO width)')
axs[1,1].set_ylim((0,11))
axs[1,2].set_ylabel('minimised energy / (hbar omega)')

fig.suptitle('Parameter minima as a function of e_dd (D=5.e-3,a_ratio=0.01,N=1.e6)')

plt.show()