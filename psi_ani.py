import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from eGPE_1d import *

### Animate wavefunction from parameters ###

params = np.genfromtxt('params.csv',delimiter=',')

fig, ax = plt.subplots()
step, zs, ks, k_range, L = set_mesh(z_len)
def animate(i):
    ax.clear()
    zs = np.linspace(-10,10,1000)
    psisq = psi_0(zs,*params[-i-1][2:])**2
    ax.plot(zs,psisq/(np.sum(psisq)*step))
    ax.set_ylim((0,7))

ani = animation.FuncAnimation(fig, animate, 20,interval=500)
plt.show()