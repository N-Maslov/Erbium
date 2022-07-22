import numpy as np
import scipy.special as sc
import ftransformer as ft
from scipy.optimize import minimize
import warnings
import cProfile
warnings.filterwarnings('error')

# set parameters
n = 1.0e19      # mean density
L = 0.1         # length
a_s = 3.0e-9    # contact length
e_dd = 1        # interaction ratio
m = 167.259*1.66e-27
hbar = 1.055e-34# duh
omegas = 2*np.pi*np.array([150,150,0]) # steepness of potential
RES = 2048      # array length for integral and FFT, pfastest w/ power of 2

# set wavefunction
def psi_0(z):
    return n**0.5*z**0

# preliminary calculation
g_s = 4*np.pi*hbar**2/m * a_s
g_dd = g_s*e_dd
gam_QF = 32/3 * g_s *(a_s**3/np.pi)**0.5 * (1+3/2*e_dd)
N = n * L # number of particles
step = L / RES # z increment

zs = np.linspace(-L/2,L/2,RES,endpoint=False)
psis = psi_0(zs)
psisq = np.abs(psis)**2
F_psi = ft.f_x4m(RES,L,psis)[0]
F_psi_sq, ks = ft.f_x4m(RES,L,psisq)

def particle_energy(eta:float,l:float):
    """Calculate per-particle energy given wavefunction parameters:
     - anisotropy eta
     - mean width l.
     Does not take vector inputs! Use energies_mat function for that."""
    # preliminary calc
    gam_sig = 2/(5*np.pi*1.5*l**3)
    g_QF = gam_QF*gam_sig
    # initialise with perpendicular energy contribution
    val = hbar**2/(4*m*l**2)*(eta+1/eta) + m*l**2/4*(omegas[0]**2/eta+omegas[1]**2*eta)
    # get kinetic energies for each point
    KE_contribs = hbar**2/(2*m) * ft.inv_f_x4m(RES,L,F_psi*ks**2)[0].real
    Phis = Phi(eta,l)
    index = 0
    for z in zs:
        psi = psi_0(z) # wavefunction value to put into integrand
        val += step/N*(
            np.conjugate(psi) * (
            (2/5*g_QF*np.abs(psi)**3 + 1/2*Phis[index].real + 1/2*m*omegas[2]**2*z**2)*psi
            +KE_contribs[index]
            )   
        ) # get integrand at each point
        index+=1
    return val


def Phi(eta:float,l:float):
    # drops redundant corresponding z's on return 
    return ft.inv_f_x4m(RES,ks[-1]-ks[0],U_sig(ks,eta,l)*F_psi_sq)[0]

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

def energies_mat(etas:np.ndarray,ls:np.ndarray):
    """Returns matrix of per-particle energies, each of which corresponds
    to a separate (eta,l) pair."""
    vals = np.zeros((len(etas),len(ls)))
    for i in range(len(etas)):
        for j in range(len(ls)):
            vals[i,j] = particle_energy(etas[i],ls[j])
    return vals

energy_func = lambda x: particle_energy(x[0],x[1])*1.e26
bnds = ((0.01,None),(1.e-9,None))
res = minimize(energy_func,(1.1,0.001),bounds=bnds)
print(res.x)
#cProfile.run('')