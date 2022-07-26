import numpy as np
import scipy.special as sc
import ftransformer as ft
from scipy.optimize import minimize
import warnings
import cProfile
warnings.filterwarnings('error')

# set parameters
n = 1.0e19      # mean density
L =  6.79223076e-02        # length
a_s = 3.0e-9    # contact length
e_dd = 1        # interaction ratio
m = 166*1.66e-27
hbar = 1.055e-34# duh
omegas = 2*np.pi*np.array([150,150,0]) # steepness of potential
RES = 2048      # array length for integral and FFT, fastest w/ power of 2

# preliminary calculation
g_s = 4*np.pi*hbar**2/m * a_s
g_dd = g_s*e_dd
gam_QF = 32/3 * g_s *(a_s**3/np.pi)**0.5 * (1+3/2*e_dd)
step = L / RES # z increment

zs = np.linspace(-L/2,L/2,RES,endpoint=False)



def particle_energy(psi_args,psi_0):
    """Calculate per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take vector inputs! Use energies_mat function for that."""

    eta = psi_args[0]
    l = psi_args[1]
    psi_0_args = psi_args[2:]

    # wavefunction calc
    try:
        psis = psi_0(zs,*psi_0_args)
    except TypeError: # if psi_0 has 0 free parameters
        psis = psi_0(zs)
    psisq = np.abs(psis)**2
    F_psi = ft.f_x4m(RES,L,psis)[0]
    F_psi_sq, ks = ft.f_x4m(RES,L,psisq)
    k_range = ks[-1]-2*ks[0]+ks[1]

    # Number of particles:
    N = np.sum(psisq)*step

    # preliminary calc of constants
    gam_sig = 2/(5*np.pi*1.5*l**3)
    g_QF = gam_QF*gam_sig

    # initialise with perpendicular energy contribution
    val = hbar**2/(4*m*l**2)*(eta+1/eta) + m*l**2/4*(omegas[0]**2/eta+omegas[1]**2*eta)*N/step
    # get kinetic energies for each point
    KE_contribs = hbar**2/(2*m) * ft.inv_f_x4m(RES,k_range,F_psi*ks**2)[0].real / (2*np.pi)**2

    #Phis = Phi(eta,l,F_psi_sq,ks)
    Phis = ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq)[0]
    index = 0
    for z in zs:
        psi = psis[index] # wavefunction value to put into integrand
        val += np.conjugate(psi) * (
                (2/5*g_QF*np.abs(psi)**3 + 1/2*Phis[index].real + 1/2*m*omegas[2]**2*z**2)*psi
                +KE_contribs[index]
                )  # get integrand at each point
        index+=1
    return val*step/N


def Phi(eta:float,l:float,F_psi_sq,ks):
    # drops redundant corresponding z's on return 
    return #ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq)[0]

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

# set wavefunction
def psi_0(z,theta,L_psi):
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return n**0.5 * (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/L_psi))

print(particle_energy((1.07552157e+00, 5.61507012e-04, 6.39240335e-02, 6.79223076e-02),psi_0))

### MINIMISATION ###

energy_func = lambda x,psi_0: particle_energy(x,psi_0)*1.e30
# Set bounds for eta, l, additional psi arguments
bnds = ((0.01,None),(1.e-9,None),(0,0.616),(1.e-9,L))
# Set initial guess
x_0 = (1.2,0.01,0.2,0.01)
res = minimize(energy_func,x_0,bounds=bnds,args=(psi_0))
print(res.x)
#cProfile.run('res = minimize(energy_func,(1.1,0.001),bounds=bnds,args=(psi_0))')"""