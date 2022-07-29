import numpy as np
import scipy.special as sc
import ftransformer as ft
from scipy.optimize import minimize
from scipy.fftpack import diff
import warnings
import cProfile
from copy import deepcopy
warnings.filterwarnings('error')

# set parameters
A = 2.e-4       # overall interaction strength
e_dd = 1        # dipole interaction strength
a_ratio = 0.1   # trap aspect ratio, omega_z / omega_R
N = 5.e5        # number of particles

# computational preferences
z_len = 1/a_ratio**0.5  # characteristic length of z-trap
L = 100 * z_len         # length of mesh (units of characteristic L of z-trap)
RES = 2048              # array length for integral and FFT, fastest w/ power of 2

# preliminary calculation
step = L / RES # z increment
zs = np.linspace(-L/2,L/2,RES,endpoint=False)
ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
k_range = ks[-1]-2*ks[0]+ks[1]
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
    try:
        psis = psi_0(zs,*psi_0_args)
    except TypeError: # if psi_0 has 0 free parameters
        psis = psi_0(zs)
    psisq = np.abs(psis)**2
    F_psi_sq = ft.f_x4m(RES,L,psisq)

    # initialise with perpendicular energy contribution
    val = 0.25*(eta+1/eta)*(l**2+1/l**2) / step

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
    return val*step


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
def psi_0(z,sigma):
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2))

cProfile.run('particle_energy((1.24,1.79,z_len*1.94),psi_0)')