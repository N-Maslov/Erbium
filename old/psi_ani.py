###################################################
#    Animates wavefunction evolution with e_dd    #
###################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from eGPE_1d import set_mesh

'''def psi_0(z,sigma,theta,period):
    """Muodified definition of longitudinal wavefunction to prevent 
    division by zero errors when modulation absent (period 0)"""
    if period < 0.001:
        return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2))
    else:
        return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2)) *\
        (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))'''
def psi_0(z,sigma,theta,period):
    x = z/sigma
    if period < 0.001:
        return 1/(1 + 1/20*x**2 + 21/800*x**4 + 0.056*x**6)**10
    else:
        return 1/(1 + 1/20*x**2 + 21/800*x**4 + 0.056*x**6)**10 *\
        (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))

def animate(i,params,xvalslist):
    ax.clear()
    ax.grid()
    #zs = np.linspace(-15,15,1000)
    psisq = psi_0(zs,*params[i][2:])**2
    line, = ax.plot(zs,psisq/(np.sum(psisq)*step))
    ax.text(0.05,0.8,
        'e_dd = '+"{:.3f}".format(xvalslist[i]),
    fontsize = 'large',transform=ax.transAxes)
    #ax.set_ylim((0,10))
    return line

def ani_generator(fold='run_0\\',indx=0,save=False,frame_len=50):
    """Produces an animation of a constant a_ratio slice evolving with e_dd.
    Fold: folder containing the data, indx is the index of the y value (a_ratio).
    if save, will be saved as a gif. Frame_len in ms."""
    # Populate arrays with parameters read from file
    outMats = []
    xvalslist = np.loadtxt(fold+'e_vals.csv',delimiter=',')
    yvalslist = np.loadtxt(fold+'a_vals.csv',delimiter=',')
    names = [fold+'outputs_'+str(i)+'.csv' for i in range(6)]
    for name in names:
        outMats.append(np.loadtxt(name,delimiter=','))
    params = np.concatenate((
        [outMats[0][indx,:]],[outMats[1][indx,:]],[outMats[2][indx,:]],
        [outMats[3][indx,:]],[outMats[4][indx,:]]),axis=0).T
    
    ax.set_ylim((0,10))
    fig.suptitle(f'aspect ratio = '+"{:.3f}".format(yvalslist[indx]))
    ani = animation.FuncAnimation(fig, animate, range(len(xvalslist)),
        interval=frame_len,fargs=(params,xvalslist))
    if save:
        ani.save('psi_evolution.gif','pillow')
    plt.show()

fig, ax = plt.subplots()
step, zs = set_mesh(60)[:2]

if __name__ == '__main__':
    ani_generator(fold='run_0\\',indx=0,frame_len=300)