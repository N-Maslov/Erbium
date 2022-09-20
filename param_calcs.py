'''
### PARAMETER CALCULATIONS ###
The functions here are called after finding energy minima, to determine which number of droplets
has the lowest energy, and the contrast and decay time of a given wavefunction.'''

import numpy as np
from scipy.signal import find_peaks
from copy import copy

def get_contrast(psisq:np.ndarray,RES:int,n:int)->float:
    '''Returns wavefunction contrast in the centre of the integration region.
    Inputs:
    - psisq: array of values of the wavefunction squared
    - RES: grid resolution
    - n: number of droplets
    Returns:
    - contrast: (difference/sum) of maximum and minimum of probability density'''

    if n == 1:
        # single droplet has no modulation so contrast = 0
        return 0

    if n%2 == 0:
        # for even numbers of droplets, take minimum at z=0, and maximum at global max
        psisq_min = psisq[int(RES/2)]
        psisq_max = np.amax(psisq)
        
    else:
        # for odd numbers of droplets, take maximum in the centre,
        # and minimum as the first-order minimum from it
        psisq_max = psisq[int(RES/2)]
        min_positions = find_peaks(-psisq)[0]
        min_vals = np.array([psisq[x] for x in min_positions])

        try:
            psisq_min = np.amax(min_vals)
        except ValueError:
            # if multiple droplets are close enough together, they merge into a single
            # unit so there is no minimum and therefore contrast is 0.
            return 0

    return (psisq_max-psisq_min)/(psisq_max+psisq_min)

def get_lifetime(psisq:np.ndarray,step:float,l:float,epsilon_dd:float,
    freq:float,N_atoms:float)->float:
    '''Returns estimate of 3-body loss decay time based on second moment of density
    and fitting of three-body-loss coefficient to e_dd from lab data.
    Inputs:
    - psisq: array of values of wavefunction squared
    - step: separation between mesh points in real space
    - l: mean transverse width of wavefunction
    - epsilon_dd: a_dd / a_s
    - freq: transverse trap frequency (Hz)
    - N_atoms: number of atoms in the trap
    Returns:
    - log of the characteristic exponential decay time in seconds'''

    # work out 3-body-loss coefficient from fit to Milan's data
    loss_coeff = (5.75*epsilon_dd - 6.03) * 4.43e-10 * freq**3 * N_atoms**2

    # numerically integrate the z-wavefunction. (x and y done analytically)
    # and use to calculate the decay time
    return np.log10(1/(loss_coeff*np.sum(psisq**3)*step)*np.pi**2*l**4*3)
    
def get_minima(params_arrays:tuple,energies_arrays:tuple,ns:np.ndarray)->tuple:
    '''Takes data from runs for different number of droplets and selects those that
    correspond to the number of droplets with the minimum energy.
    Inputs:
    - params_arrays: tuple of (2D; parameters and different e_dd) numpy arrays
        of parameters corresponding to different numbers of droplets.
        i.e. params_arrays[n][i,j] is for ith point along e_dd sweep, jth parameter, for n-1 droplets
    - energies_arrays: tuple of (1D: for different e_dd) numpy arrays of energies
        corresponding to different numbers of droplets.
        i.e. energies_arrays[n][i] is for ith point along e_dd sweep, for n-1 droplets
    Returns:
    - min_energies: 1D array of lowest energies found across different droplet numbers
    - min_params: 2D array of corresponding parameters (jth parameter for ith e_dd point at min_params[i,j])
    - min_ns: 1D array of corresponding numbers of droplets'''

    # create energies array to compare to and replace term if lower
    min_energies = 1000*np.ones_like(energies_arrays[0],dtype=float)

    # create corresponding arrays for parameters and droplet numbers that minimise the energy
    min_params = np.empty_like(energies_arrays[0],dtype=object)
    min_ns = np.empty_like(energies_arrays[0],dtype=int)

    # iterate through each individual number of droplets and the corresponding data
    for run,energies in enumerate(energies_arrays):
        # generate boolean array of terms to replace
        replace = (energies < min_energies)

        # iterate through values taken at different e_dd
        for pos in range(len(energies)):
            if replace[pos]:
                # replace current stored values of energy, parameters and n if new energy
                # found to be lower.
                min_energies[pos] = copy(energies[pos])
                min_params[pos] = copy(params_arrays[run][pos])
                min_ns[pos] = ns[run]

    return min_energies,min_params,min_ns