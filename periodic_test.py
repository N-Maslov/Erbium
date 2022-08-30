####################################
# Periodic version for testing 
####################################
import numpy as np
import scipy.special as sc
import old.ftransformer as ft
from scipy.optimize import minimize
from scipy.fftpack import diff
import warnings
import matplotlib.pyplot as plt
import cProfile
from copy import deepcopy
warnings.filterwarnings('error')

# set parameters
D = 0.010798       # overall interaction strength (FIXED) omega = 150 gives 5.46e-4 for erbium, 0.0108 for Dy
e_dd = 1.37        # dipole interaction strength
n = 1602           # denisty 1602

# computational preferences
RES = 2**10             # array length for integral and FFT, fastest w/ power of 2, must be EVEN

# preliminary calculation
def set_mesh(L):
    step = L / RES # z increment
    zs = np.linspace(-L/2,L/2,RES,endpoint=False)
    ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
    k_range = ks[-1]-2*ks[0]+ks[1]
    return step, zs, ks, k_range

def get_coeff(D,e_dd,N):
    A = D/e_dd
    pref_inter = A*N # prefactor for interaction term
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term
    return pref_inter, pref_QF


thetconstr = np.arctan(1/2**0.5)

def particle_energy(psi_args,psi_0):
    """Calculate per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take numpy vector inputs!
    modify with conjugate, abs to allow complex psi."""

    eta = psi_args[0]
    l = psi_args[1]
    theta,period = psi_args[2:]

    # wavefunction calc
    step, zs, ks, k_range = set_mesh(period)
    N = n*period
    pref_inter, pref_QF = get_coeff(D,e_dd,N)


    psis = psi_0(zs,theta,period)
    psisq = psis**2
    F_psi_sq = ft.f_x4m(RES,period,psisq)

    KE_contribs = -0.5 * diff(psis,2,period)
    Phis = np.real(ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq))

    return 0.25*(eta+1/eta)*(l**2+1/l**2) + step*(psisq @ (
        pref_QF*np.abs(psis/l)**3 + pref_inter/l**2*Phis
    ) + psis@KE_contribs)


def U_sig(ks,eta,l):
    """Calculate approximation function for 2D fourier transform"""
    Q_sqs = 1/2*eta**0.5*(ks*l)**2
    Q_sqs = np.where(Q_sqs<703,Q_sqs,703*np.ones_like(ks,dtype=float))
    # low value limit: ks is 0 at RES/2
    try:
        Q_sqs[int(RES/2)] = 1.e-18
    except IndexError:
        pass
    # normal calculation
    numerat = np.where(Q_sqs<703,
        3*(Q_sqs*np.exp(Q_sqs)*sc.expi(-Q_sqs)+1),3/Q_sqs)
    # high value limit automatically zero now
    return 1+e_dd * (numerat/(1+eta)-1)

def thet_wrap(thetas):
    thetas = thetas % (2*thetconstr)
    return np.where(thetas>thetconstr, thetas-2*thetconstr,thetas)

# set wavefunction
def psi_0(z,theta,period):
    theta = thet_wrap(theta)
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))/period**0.5

contrast = lambda theta: (2**1.5 * np.sin(2*theta))/(3-np.cos(2*theta))

# testing ground
'''eta = 1.47
l = 1.32
thet = 0.4
lam = 1.77

Lambda = 1/32*(90*np.cos(thet)-55*np.cos(3*thet)-3*np.cos(5*thet))
bigbit = U_sig(np.array([1.e-18]),eta,l)[0] + np.sin(2*thet)**2*U_sig(np.array([2*np.pi/lam]),eta,l)[0]\
            +1/2*np.sin(thet)**4*U_sig(np.array([4*np.pi/lam]),eta,l)[0]

#print(2*np.pi**2*np.sin(thet)**2/lam**2) # ke
A = D/e_dd
print(512/(75*np.pi)*A**2.5*(1+1.5*e_dd**2)/l**3*n**1.5*Lambda) #qf
print(n*A/l**2*bigbit)
print(particle_energy((eta,l,thet,lam),psi_0)) '''









############################################
x_0 = 4,1,thetconstr,4
bnds = (0.9,None),(0.00001,None),(None,None),(1/RES,None)
e_vals = np.linspace(1.4897,1.4217,300)
min_etas = np.zeros_like(e_vals, dtype=float)
min_ls = deepcopy(min_etas)
min_thetas = deepcopy(min_etas)
min_periods = deepcopy(min_etas)
min_energies = deepcopy(min_etas)
len_res = deepcopy(min_etas)
for i,e_dd in enumerate(e_vals):
    res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')

    min_etas[i] = res.x[0]
    min_ls[i] = res.x[1]
    min_thetas[i] = res.x[2]
    min_periods[i]= res.x[3]
    min_energies[i] = particle_energy((res.x),psi_0)
    #len_res[i] = 1/RES

    x_0 = res.x*np.random.normal(1,0.01,(4))
    print(i)


### PLOTTING ###
fig, axs = plt.subplots(2,2)
axs[0,0].plot(130.8/e_vals,min_etas,'.')
axs[0,1].plot(130.8/e_vals,min_ls*0.641,'.',label='l')
axs[0,1].plot(130.8/e_vals,min_periods*0.641,'.',label='L')

axs[1,0].plot(130.8/e_vals,contrast(np.abs(thet_wrap(min_thetas))),'.')
axs[1,1].plot(130.8/e_vals,min_energies,'.')

for ax in np.reshape(axs,(4)):
    ax.set_xlabel('a_s/a_0')
axs[0,0].set_ylabel('obliquity eta')
axs[0,1].set_ylabel('width / (QHO width)')
axs[0,1].legend()
axs[1,0].set_ylabel('contrast')
axs[1,1].set_ylabel('energy')


fig.suptitle(f'Modulation only - parameter minima as a function of e_dd (D={D},n={n},res={RES})')  

print(res.x)
plt.show()
#"""