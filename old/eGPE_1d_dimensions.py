import numpy as np
import scipy.special as sc
import old.ftransformer as ft
from scipy.optimize import minimize
from scipy.fftpack import diff
import warnings
import cProfile
from copy import deepcopy
warnings.filterwarnings('error')

# set parameters
m = 166*1.66e-27
hbar = 1.055e-34# duh
omegas = 2*np.pi*np.array([150,150,15]) # steepness of potential

# for QHO treatment
char_l_xy = np.sqrt(hbar/(m*np.sqrt(omegas[0]*omegas[1])))
char_l_z = np.sqrt(hbar/(m*omegas[2]))
char_e = hbar*np.sum(omegas)/2

n = 2.5e9      # mean density
L = char_l_z*1.e2        # length
a_s = 5.97e-11*80    # contact length
e_dd = 1        # interaction ratio
RES = 2048      # array length for integral and FFT, fastest w/ power of 2

# preliminary calculation
g_s = 4*np.pi*hbar**2/m * a_s
g_dd = g_s*e_dd
gam_QF = 32/3 * g_s *(a_s**3/np.pi)**0.5 * (1+3/2*e_dd**2)
step = L / RES # z increment

zs = np.linspace(-L/2,L/2,RES,endpoint=False)
ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
k_range = ks[-1]-2*ks[0]+ks[1]


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

    # Number of particles:
    N = np.sum(psisq)*step

    # preliminary calc of constants
    gam_sig = 2/(5*np.pi**1.5*l**3)
    g_QF = gam_QF*gam_sig

    # initialise with perpendicular energy contribution
    val = 0*(m*l**2/4*(omegas[0]**2/eta+omegas[1]**2*eta)+hbar**2/(4*m*l**2)*(eta+1/eta))*N/step
    # get kinetic energies for each point
    KE_contribs = -hbar**2/(2*m) * diff(psis,2,L)
    Phis = ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq)
    index = 0
    for z in zs:
        psi = psis[index] # wavefunction value to put into integrand
        val += np.conjugate(psi) * (
                (2/5*g_QF*np.abs(psi)**3 + 0*1/2*Phis[index].real + 0*1/2*m*omegas[2]**2*z**2)*psi
                +0*KE_contribs[index]
                )  # get integrand at each point
        index+=1
    return val*step/N


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
    return g_s/(2*np.pi*l**2) + g_dd/(2*np.pi*l**2) * (numerat/(1+eta)-1)


# set wavefunction
def psi_0(z,sigma):
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return (n*L/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2))
    #return n**0.5 * (np.cos(theta)+2**0.5*np.sin(theta)*np.cos(2*np.pi*z/L_psi))

#print(particle_energy((1.24, char_l_xy*1.79, char_l_z*1.94),psi_0)/(hbar*omegas[0]))
#cProfile.run('particle_energy((1.24, char_l_xy*1.79, char_l_z*1.94),psi_0)/(hbar*omegas[0])')
### MINIMSER ###
def energy_func(x,psi):
    # inputs rescaled by characterstic lengths to be O(1)
    rescaled_vect = (x[0],x[1]*char_l_xy,x[2]*char_l_z)
    return particle_energy(rescaled_vect,psi)/char_e

# Set bounds for eta, l, additional psi arguments
bnds = ((0.01,None),(1.e-9,None),(1.e-9,None))
# Set initial guess
x_0 = (1, 1, 1)

#res = minimize(energy_func,x_0,bounds=bnds,args=(psi_0))