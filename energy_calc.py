import numpy as np
from scipy.special import expi
from scipy.fftpack import diff
from typing import Callable

def f_x4m(N:int,L:float,in_vect:np.ndarray) -> np.ndarray:
    """Approximates input as periodic sampled input.
    Input vector has N entries between +-L/2.
    Returns Fourier transformed array centred on zero to best match FT definition,
    and corresponding k's."""
    ks = 2*np.pi*np.fft.fftfreq(N,L/N)
    x4m = L/N * np.exp(ks*L/2*1j) * np.fft.fft(in_vect)
    x4m = np.fft.fftshift(x4m)
    return x4m

def inv_f_x4m(N:int,k_range:float,in_vect:np.ndarray) -> np.ndarray:
    """Inverse fourier transform. Designed for compatibility with f_x4m."""
    inv_input = in_vect[::-1]
    x4m = f_x4m(N,k_range,np.concatenate(([0],inv_input[:-1])))
    return 1/(2*np.pi)*x4m

def particle_energy(psi_args:tuple,psi_0:Callable,mesh_args:tuple,RES:int,
    e_dd:float,a_ratio:float,pref_inter:float,pref_QF:float) -> float:
    """Calculate dimensionless per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take numpy vector inputs!
    modify with conjugate, abs to allow complex psi.
    Returns energy."""
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
    """Calculate approximation function for 2D fourier transform
    (intermediate calculation for particle_energy)"""
    Q_sqs = 1/2*eta**0.5*(ks*l)**2
    Q_sqs = np.where(Q_sqs<703,Q_sqs,703*np.ones_like(ks,dtype=float))
    # low value limit: ks is 0 at RES/2
    Q_sqs[int(RES/2)] = 1.e-18
    # normal calculation
    numerat = np.where(Q_sqs<703,
        3*(Q_sqs*np.exp(Q_sqs)*expi(-Q_sqs)+1),3/Q_sqs)
    # high value limit automatically zero now
    return 1+e_dd * (numerat/(1+eta)-1)

# individual contributions to energy
def transv_KE(eta,l):
    return 0.25*(eta+1/eta)/l**2

def transv_PE(eta,l):
    return 0.25*(eta+1/eta)*l**2

def longi_KE(psis,step,L):
    KE_contribs = -0.5 * diff(psis,2,L)
    return step*psis@KE_contribs

def QF(psis,psisq,step,l,pref_QF):
    return step*pref_QF*psisq@np.abs(psis/l)**3

def interaction(psisq,step,ks,k_range,L,eta,l,pref_inter,e_dd,RES):
    F_psi_sq = f_x4m(RES,L,psisq)
    Phis = np.real(inv_f_x4m(RES,k_range,U_sig(ks,eta,l,e_dd,RES)*F_psi_sq))
    return step*pref_inter/l**2 * psisq@Phis

def longi_PE(psisq,step,zs,a_ratio):
    return 1/2*step*a_ratio**2*psisq@zs**2

energy_funcs = interaction,QF,transv_KE,transv_PE,longi_KE,longi_PE