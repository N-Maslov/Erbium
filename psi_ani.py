import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from eGPE_1d import set_mesh,z_len
def psi_0(z,sigma,theta,period):
    """Must be of form psi_0(z,arg1, arg2, ...)
    Returns value of wavefunction given coordinate and psi_z parameters"""
    if period < 0.001:
        return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2))
    else:
        return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2)) *\
        (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))

### Animate wavefunction from parameters ###
outMats = []
fold = 'run_3_600x200\\' # blank for current dir, don't forget \\
xvalslist = np.loadtxt(fold+'e_vals.csv',delimiter=',')
yvalslist = np.loadtxt(fold+'a_vals.csv',delimiter=',')

names = [fold+'outputs_'+str(i)+'.csv' for i in range(6)]
for i, name in enumerate(names):
    outMats.append(np.loadtxt(name,delimiter=','))
indx = 84 #120,95,90,0, 84

#params = np.genfromtxt('params.csv',delimiter=',')
params = np.concatenate((
        [outMats[0][indx,:]],[outMats[1][indx,:]],[outMats[2][indx,:]],[outMats[3][indx,:]],[outMats[4][indx,:]]),axis=0).T
fig, ax = plt.subplots()
step, zs, ks, k_range, L = set_mesh(z_len)
def animate(i):
    ax.clear()
    zs = np.linspace(-150,150,1000)
    psisq = psi_0(zs,*params[i][2:])**2
    ax.plot(zs,psisq/(np.sum(psisq)*step))
    ax.set_ylim((0,5))
    ax.grid()
    ax.text(-20,4,
    'e_dd = '+"{:.3f}".format(xvalslist[i])+'\naspect ratio = '+"{:.2f}".format(yvalslist[indx]),
    fontsize = 'large')

ani = animation.FuncAnimation(fig, animate, range(500,600),interval=10)
#ani.save('psi_evolution.gif','pillow')
plt.show()