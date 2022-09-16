import numpy as np
from scipy.signal import find_peaks
from copy import copy
import setup

def get_contrast(n,params):
    '''Returns wavefunction contrast in centre for n droplets with parameters array params'''
    if n == 1:
        return 0
    # set integration range
    L_set = 40*params[2]
    if n%2 == 0: # even droplets
        L_set += (n-1)*params[3]
    else: # odd droplets
        L_set += n*params[3]

    func = setup.funcs[n-1]
    zs = setup.set_mesh(L_set)[1]
    psisq = func(zs,*params[2:])

    if n%2 == 0:
        psisq_min = func(0,*params[2:])
        psisq_max = np.amax(psisq)
    else:
        psisq_max = func(0,*params[2:])
        min_positions = find_peaks(-psisq)[0]
        min_vals = np.array([psisq[x] for x in min_positions])
        try:
            psisq_min = np.amax(min_vals)
        except ValueError: # for multiple droplets without minima
            return 0
    return (psisq_max-psisq_min)/(psisq_max+psisq_min)

def loss_coeff(epsilon,freq,N_atoms):
    '''Calculates dimensionless three-body-loss coefficient k N**2/L**6'''
    return (5.75*epsilon - 6.03) * 4.43e-10 * freq**3 * N_atoms**2

def get_lifetime(n,params,k=1):
    '''Returns estimate of 3-body loss decay time based on second moment of density'''
    # set integration range
    L_set = 40*params[2]
    if n%2 == 0: # even droplets
        L_set += (n-1)*params[3]
    elif n>1: # odd droplets
        L_set += n*params[3]

    func = setup.funcs[n-1]
    step,zs = setup.set_mesh(L_set)[:2]
    psisq = func(zs,*params[2:])
    N_corr = np.sum(psisq)*step
    psisq = psisq/N_corr

    # return log of decay time
    return np.log10(1/(k*np.sum(psisq**3)*step)*np.pi**2*params[1]**4*3)
    
def get_minima(params_arrays,energies_arrays,ns):
    min_energies = 1000*np.ones_like(energies_arrays[0],dtype=float)
    min_params = np.empty_like(energies_arrays[0],dtype=object)
    min_ns = np.empty_like(energies_arrays[0],dtype=int)

    for run,energies in enumerate(energies_arrays):
        replace = (energies < min_energies)
        for pos in range(len(energies)):
            if replace[pos]:
                min_energies[pos] = copy(energies[pos])
                min_params[pos] = copy(params_arrays[run][pos])
                min_ns[pos] = ns[run]

    return min_energies,min_params,min_ns
