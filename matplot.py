#############################################
#    Plots 2D matrix or a 1D slice of it
##############################################

import matplotlib.pyplot as plt
import numpy as np
from eGPE_1d import contrast,get_lifetimes,plot_1d,decorate_1d

mode = '2d'

# load data
fold = 'run_5_600x200\\' # blank for current dir, don't forget \\
names = [fold+'outputs_'+str(i)+'.csv' for i in range(6)]
outMats = []
xvalslist = np.loadtxt(fold+'e_vals.csv',delimiter=',')
yvalslist = np.loadtxt(fold+'a_vals.csv',delimiter=',')
for i, name in enumerate(names):
    outMats.append(np.loadtxt(name,delimiter=','))
lifetimes = np.loadtxt(fold+'lifetimes.csv',delimiter=',')

### temporary area
if mode == 'testing':
    outMat = np.stack(outMats,axis=2)
    lifetimes = get_lifetimes(outMat)
    np.savetxt(fold+'lifetimes.csv',lifetimes,delimiter=',')


# 2D plot
elif mode == '2d':
    fig, ax = plt.subplots()
    #ax.imshow(outMats[2])
    #ax.imshow((contrast(outMats[3])))
    #ax.imshow(np.transpose(yvalslist**(0.5*1.67)*np.transpose(outMats[2])))
    ax.imshow(np.log(lifetimes))

    nticksx = 11
    nticksy = nticksx

    ticksx = np.linspace(0,len(xvalslist)-1,nticksx)
    ticksy = np.linspace(0,len(yvalslist)-1,nticksy)
    tlabelsx = np.linspace(xvalslist[0],xvalslist[-1],nticksx)
    tlabelsy = np.linspace(yvalslist[0],yvalslist[-1],nticksy)

    ax.set_xticks(ticksx)
    ax.set_yticks(ticksy)
    ax.set_xticklabels(["{:.2f}".format(label) for label in tlabelsx])
    ax.set_yticklabels(["{:.2f}".format(label) for label in tlabelsy])

    ax.set_xlabel('e_dd')
    ax.set_ylabel('omega_z/omega_r')
    ax.set_aspect(7)
    #ax.set_title('droplet width / longitudinal width')


# slice plot
elif mode == '1d':
    indx = 55
    a_ratio = yvalslist[indx]
    fig,axs=plt.subplots(2,3)
    plot_1d(axs,np.concatenate((
        [outMats[0][indx,:]],[outMats[1][indx,:]],[outMats[2][indx,:]],[outMats[3][indx,:]],[outMats[4][indx,:]]),axis=0).T,
        outMats[5][indx,:],xvalslist)
    decorate_1d(fig,axs,a_ratio)

plt.show()