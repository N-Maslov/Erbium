'''
### SETUP ###
allows modification of the number of grid points to integrate over,
the functional form of the longitudinal wavefunctions (eg for testing more droplets),
and the initial guesses for the various parameters at the lowest values of e_dd in the sweep.
Functions set_mesh, get_coeff, and get_D should not be modified.
'''

import numpy as np

# Set number of points for numerical integration
# must be EVEN. Ideally a power of 2 for highest speed in FFT. Default 2**12.
RES = 2**12

def set_mesh(L:float) -> tuple:
    '''Determines computational constants for desired integration length L.
    Returns: 
    - step: separation between points along mesh
    - zs: array of all z-points along the grid
    - ks: array of points in Fourier space
    - k_range: total integration length in Fourier space
    - L total integration length in real space (duplicate)
    gets computational constants (step, zs, ks, k_range, l) for given grid size L'''

    step = L / RES # z increment
    zs = np.linspace(-L/2,L/2,RES,endpoint=False)
    ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
    k_range = ks[-1]-2*ks[0]+ks[1]
    return step, zs, ks, k_range, L

def get_coeff(D:float,e_dd:float,N:float) -> tuple:
    '''Determines prefactors for interaction and quantum fluctuation terms
    Inputs: 
    - interaction strength D = a_dd in units of characteristic transverse length
    - e_dd
    - total number of atoms N
    Returns:
    - pref_inter: prefactor for interaction term (d-d and contact)
    - pref_QF: prefactor for quantum fluctuation term'''

    A = D/e_dd # dimensionless scattering length
    pref_inter = A*N # prefactor for interaction term
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term
    return pref_inter, pref_QF

def get_D(f:float,mu=7,m=166) -> float:
    '''Returns dimensionless interaction strength D=a_dd/l for given transverse trap
    frequency f, and optionally magnetic moment (in mu_b) and atomic mass, default to Erbium.'''
    
    return 4.257817572e-9 * np.sqrt(m**3*f)*mu**2

# set wavefunctions for different numbers of droplets
def psi_0(z,s): # 1 droplet, non-gaussian ansatz, matches TF in centre and tapering off rate contrallable
    x = z/s
    return 1/(1 + 1/20*x**2 + 21/800*x**4 + 0.056*x**6)**10
psi_1 = lambda z,s: np.exp(-z**2/(2*s**2))
psi_2 = lambda z,s,w: np.exp(-(z-w/2)**2/(2*s**2)) + np.exp(-(z+w/2)**2/(2*s**2))
psi_3 = lambda z,s,w,h_1: h_1*(np.exp(-(z-w)**2/(2*s**2)) + np.exp(-(z+w)**2/(2*s**2))) + psi_1(z,s)
psi_4 = lambda z,s,w,h_1: h_1*(np.exp(-(z-3*w/2)**2/(2*s**2)) + np.exp(-(z+3*w/2)**2/(2*s**2))) + psi_2(z,s,w)
psi_5 = lambda z,s,w,h_1,h_2: h_2*(np.exp(-(z-2*w)**2/(2*s**2)) + np.exp(-(z+2*w)**2/(2*s**2))) + psi_3(z,s,w,h_1)
psi_6 = lambda z,s,w,h_1,h_2: h_2*(np.exp(-(z-5*w/2)**2/(2*s**2)) + np.exp(-(z+5*w/2)**2/(2*s**2))) + psi_4(z,s,w,h_1)

# array of wavefunctions, used in other scripts.
# must be in order of number of droplets and starting from 1 increasing by 1 each time
funcs = (psi_0,psi_2,psi_3,psi_4,psi_5)

# initial guesses for minimisation, starting at lowest value of e_dd.
# each key corresponds to its number of droplets.
init_guesses = {1:[1,1,1],
    2:[1,1,1,5],
    3:[1,1,1,5,0.9],
    4:[1,1,1,5,0.9],
    5:[1,1,1,5,0.8,0.6]}