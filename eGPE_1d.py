import numpy as np
import scipy.special as sc
import ftransformer as ft
from scipy.optimize import minimize
from scipy.fftpack import diff
import warnings
import matplotlib.pyplot as plt
import cProfile
from copy import deepcopy
warnings.filterwarnings('error')

# set parameters
D = 2.2e-3          # overall interaction strength (FIXED) 5.e-3
e_dd= 1.50          # dipole interaction strength
a_ratio = 0.2      # trap aspect ratio, omega_z / omega_R
N = 3.0e4           # number of particles 8.0e3

# computational preferences
z_len = 1/a_ratio**0.5  # characteristic length of z-trap
L = 40 * z_len         # length of mesh (units of characteristic L of z-trap)
RES = 2**10             # array length for integral and FFT, fastest w/ power of 2

# preliminary calculation
step = L / RES # z increment
zs = np.linspace(-L/2,L/2,RES,endpoint=False)
ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
k_range = ks[-1]-2*ks[0]+ks[1]
A = D/e_dd
pref_inter = A*N # prefactor for interaction term
pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term


def particle_energy(psi_args,psi_0):
    """Calculate per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take numpy vector inputs!"""

    eta = psi_args[0]
    l = psi_args[1]
    psi_0_args = psi_args[2:]

    # wavefunction calc
    psis = psi_0(zs,*psi_0_args)
    psisq = np.abs(psis)**2
    N_corr = np.sum(psisq)*step
    psis = psis/N_corr**0.5
    psisq = psisq/N_corr
    F_psi_sq = ft.f_x4m(RES,L,psisq)

    val = 0
    # get kinetic energies for each point
    KE_contribs = -0.5 * diff(psis,2,L)
    Phis = np.real(ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq))
    index = 0
    for z in zs:
        psi = psis[index] # wavefunction value to put into integrand
        val += np.conjugate(psi) * (
                (pref_QF/l**3*np.abs(psi)**3 + pref_inter/l**2*Phis[index] + 1/2*a_ratio**2*z**2)*psi
                +KE_contribs[index]
                )  # get integrand at each point
        index+=1
    return val*step + 0.25*(eta+1/eta)*(l**2+1/l**2)


def U_sig(ks:np.ndarray,eta:float,l:float):
    """Calculate approximation function for 2D fourier transform."""
    Q_sqs = (1/2**0.5 * ks *eta**0.25 * l)**2
    # calculate for limiting cases
    numerat = np.zeros_like(ks)
    for index, Q_sq in enumerate(Q_sqs):
        if Q_sq == 0:
            numerat[index] = 3.0
        else:
            try: # actual calculation
                expo = np.exp(Q_sq)
                numerat[index] = 3*(Q_sq*expo*sc.expi(-Q_sq)+1)
            except RuntimeWarning: # account for overflow error
                pass # since zero already set
    return 1+e_dd * (numerat/(1+eta)-1)


# set wavefunction
def psi_0(z,sigma,theta,period):
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2)) *\
    (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))


### MINIMISATION ###
x_0 = 1,1,z_len,0.6,0.1*z_len
bnds = (0.9,None),(0.1,None),(10*step,None),(0,0.6154),(10*step,None)
print(step)

### Plots generation
"""e_vals = np.linspace(1,2,10)
min_etas = np.zeros_like(e_vals,dtype=float)
min_ls = deepcopy(min_etas)
min_sigmas = deepcopy(min_etas)
min_thetas = deepcopy(min_etas)
min_periods = deepcopy(min_etas)
min_energies = deepcopy(min_etas)

for i,e_dd in enumerate(e_vals):
    A = D/e_dd
    pref_inter = A*N
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term 
    res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')
    min_etas[i] = res.x[0]
    min_ls[i] = res.x[1]
    min_sigmas[i] = res.x[2]
    min_thetas[i] = res.x[3]
    min_periods[i]= res.x[4]
    min_energies[i] = particle_energy((res.x),psi_0)
    print(i)


### PLOTTING ###
fig, axs = plt.subplots(2,3)
axs[0,0].plot(e_vals,min_etas)
axs[0,1].plot(e_vals,min_ls)
axs[0,2].plot(e_vals,min_sigmas)
axs[1,0].plot(e_vals,min_thetas)
axs[1,1].plot(e_vals,min_periods)
axs[1,2].plot(e_vals,min_energies)

for ax in np.reshape(axs,(6)):
    ax.set_xlabel('e_dd')
axs[0,0].set_ylabel('obliquity eta')
axs[0,1].set_ylabel('transverse width / (QHO width)')
axs[0,2].set_ylabel('longitudinal width / (QHO width)')
axs[1,0].set_ylabel('contrast parameter')
axs[1,1].set_ylabel('period / (QHO width)')
axs[1,2].set_ylabel('minimised energy / (hbar omega)')

fig.suptitle(f'Parameter minima as a function of e_dd (D={D},a_ratio={a_ratio},N={N})')  

plt.show()"""


#2D PLOT
SIZE = 20
yvalslist = np.linspace(0.02,0.3,SIZE,endpoint=True)
xvalslist = np.linspace(1.7,2,SIZE,endpoint=True)

outMat = np.zeros((SIZE,SIZE,6),dtype=float)
for i, a_ratio in enumerate(yvalslist):
    z_len = 1/a_ratio**0.5  # characteristic length of z-trap
    L = 40 * z_len         # length of mesh (units of characteristic L of z-trap)
    step = L / RES # z increment
    zs = np.linspace(-L/2,L/2,RES,endpoint=False)
    ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
    k_range = ks[-1]-2*ks[0]+ks[1]
    x_0 = 1,1,z_len,0.6,0.1*z_len
    bnds = (0.9,None),(0.1,None),(10*step,None),(0,0.6154),(10*step,None)

    for j, e_dd in enumerate(xvalslist):
        A = D/e_dd
        pref_inter = A*N # prefactor for interaction term
        pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term
        res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')
        #print(res.x)
        outMat[i][j][:-1] = res.x
        outMat[i][j][-1] = particle_energy(res.x,psi_0)
        #outMat[i][j] = np.random.rand()
        #print(j)
    print('NEW i = ',i)


names = ['outputs_'+str(i)+'.csv' for i in range(6)]
for i, name in enumerate(names):
    np.savetxt(name,outMat[:,:,i],delimiter=',')
vals = np.savetxt('vals.csv',[xvalslist,yvalslist],delimiter=',')