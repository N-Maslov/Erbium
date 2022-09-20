'''
### ENERGY CALCULATOR ###
Finds the energy given wavefunction parameters, and contains intermediate functions necessary
to do so, along with functions for individual energy terms (which are not called in the minimisation
because of the slight performance reduction.
'''
import numpy as np
import warnings
from scipy.special import expi
from scipy.fftpack import diff
from typing import Callable

# prevent raising overflowerrors for discarded terms in U_sig function
warnings.simplefilter('ignore',RuntimeWarning,112)

def f_x4m(N:int,L:float,in_vect:np.ndarray) -> np.ndarray:
    """Modified DFT that approximates the Fourier transform of a periodically-sampled
    function as closely as possible. Omega (not f) convention, 2 pi on inverse.
    Inputs:
    - N: length of input vector
    - L: length of integration region
    - in_vect: array of points to Fourier transform
    Returns:
    - x4m: Fourier-transformed array. Middle point corresponds to zero frequency"""

    ks = 2*np.pi*np.fft.fftfreq(N,L/N)
    x4m = L/N * np.exp(ks*L/2*1j) * np.fft.fft(in_vect)
    x4m = np.fft.fftshift(x4m)
    return x4m

def inv_f_x4m(N:int,k_range:float,in_vect:np.ndarray) -> np.ndarray:
    """Inverse fourier transform. Designed for compatibility with f_x4m.
    Inputs:
    - N: length of input vector
    - k_range: length of integration region in k-space
    - in_vect: vector to perform inverse transform on
    Returns:
    - inv_x4m: inverse Fourier transformed vector"""

    inv_input = in_vect[::-1]
    x4m = f_x4m(N,k_range,np.concatenate(([0],inv_input[:-1])))
    return 1/(2*np.pi)*x4m

def particle_energy(psi_args:tuple,psi_0:Callable,mesh_args:tuple,RES:int,
    e_dd:float,a_ratio:float,pref_inter:float,pref_QF:float) -> float:
    """Calculate energy-per-particle in units of hbar omega_r.
    Assumes trap elongated along z-axis, with dipoles aligned along y.
    Inputs:
    - psi_args: arguments of longitudinal wavefunction
    - psi_0: longitudinal wavefunction, in the form psi(z,*args)
    - mesh_args: tuple corresponding to outputs of set_mesh function (setup.py)
    - RES: number of points to integrate over
    - e_dd: a_dd/a_s
    - a_ratio: aspect ratio of trap = omega_z/omega_r
    - pref_inter, pref_QF: prefactors for interaction and QF terms in the Hamiltonian,
        corresponding to outputs of get_coeff function (setup.py)
    Returns:
    - energy per particle
    Constraints:
    - psi_args[:2] must be anisotropy eta and mean width l.
    - wavefunciton must be real (can be made complex with addition of some np.abs())
    - psi_args must be one-dimensional (does not support elementwise calculation for
        multiple sets of parameters simultaneously)"""

    # get integration parameters
    step, zs, ks, k_range, L = mesh_args

    # extract wavefunction parameters from supplied arguments
    eta = psi_args[0]
    l = psi_args[1]
    psi_0_args = psi_args[2:]

    # calculate and normalise wavefunction
    psis = psi_0(zs,*psi_0_args)
    psisq = psis**2
    N_corr = np.sum(psisq)*step

    psis = psis/N_corr**0.5
    psisq = psisq/N_corr
    F_psi_sq = f_x4m(RES,L,psisq)

    # get kinetic energies for each point
    KE_contribs = -0.5 * diff(psis,2,L)

    # get interaction contributions from each point
    Phis = np.real(inv_f_x4m(RES,k_range,U_sig(ks,eta,l,e_dd,RES)*F_psi_sq))

    # sum all energy terms
    return 0.25*(eta+1/eta)*(l**2+1/l**2) + step*(psisq @ (
        pref_QF*np.abs(psis/l)**3 + pref_inter/l**2*Phis + 1/2*a_ratio**2*zs**2
    ) + psis@KE_contribs)

def U_sig(ks:np.ndarray,eta:float,l:float,e_dd:float,RES:int) -> np.ndarray:
    """Calculates approximation function for 2D fourier transform
    (intermediate calculation for particle_energy)
    Inputs:
    - ks: vector of points in k-space corresponding to the wavefunction
    - eta: obliquity of transverse wavefunction
    - l: width of transverse wavefunction
    - e_dd: a_dd/a_s
    - RES: number of points to integrate over
    Returns
    - vector of U_sigma(k) for each k, as defined in Blakie 2020"""

    Q_sqs = 1/2*eta**0.5*(ks*l)**2

    # set zero term to just above zero to avoid error in expi while getting correct limit
    Q_sqs[int(RES/2)] = 1.e-18

    # work out numerator of approximation term. Use high-k expansion of funciton
    # for high values of Q_sqs that exp cannot handle (above 703)
    numerat = np.where(Q_sqs<703,3*(Q_sqs*np.exp(Q_sqs)*expi(-Q_sqs)+1),3/Q_sqs)
    
    return 1+e_dd * (numerat/(1+eta)-1)

# individual contributions to energy, called after minimisation to generate energies matrix
def transv_KE(eta:float,l:float)->float:
    return 0.25*(eta+1/eta)/l**2

def transv_PE(eta:float,l:float)->float:
    return 0.25*(eta+1/eta)*l**2

def longi_KE(psis:np.ndarray,step:float,L:float)->float:
    KE_contribs = -0.5 * diff(psis,2,L)
    return step*psis@KE_contribs

def QF(psis:np.ndarray,psisq:np.ndarray,step:float,l:float,pref_QF:float)->float:
    return step*pref_QF*psisq@np.abs(psis/l)**3

def interaction(psisq:np.ndarray,step:float,ks:np.ndarray,k_range:np.ndarray,
    L:float,eta:float,l:float,pref_inter:float,e_dd:float,RES:int)->float:
    F_psi_sq = f_x4m(RES,L,psisq)
    Phis = np.real(inv_f_x4m(RES,k_range,U_sig(ks,eta,l,e_dd,RES)*F_psi_sq))
    return step*pref_inter/l**2 * psisq@Phis

def longi_PE(psisq:np.ndarray,step:float,zs:np.ndarray,a_ratio:float)->float:
    return 1/2*step*a_ratio**2*psisq@zs**2

# package energy functions in a tuple to be imported into data_generator
energy_funcs = interaction,QF,transv_KE,transv_PE,longi_KE,longi_PE