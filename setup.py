import numpy as np
RES = 2**12

def set_mesh(L:float) -> tuple:
    '''gets computational constants (step, zs, ks, k_range, l) for given grid size L'''
    step = L / RES # z increment
    zs = np.linspace(-L/2,L/2,RES,endpoint=False)
    ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
    k_range = ks[-1]-2*ks[0]+ks[1]
    return step, zs, ks, k_range, L

def get_coeff(D:float,e_dd:float,N:float) -> tuple:
    '''gets coefficients for interaction and QF terms from parameters supplied'''
    A = D/e_dd # dimensionless scattering length
    pref_inter = A*N # prefactor for interaction term
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term
    return pref_inter, pref_QF

def get_D(f:float,mu=7,m=166) -> float:
    '''Returns dimensionless interaction strength D=a_dd/l for given frequency,
    and optionally magnetic moment (in mu_b) and mass, default to Erbium.'''
    return 4.257817572e-9 * np.sqrt(m**3*f)*mu**2

def psi_0(z,s,a):
    x = z/s
    return 1/(1 + 1/20*x**2 + 21/800*x**4 + (a*x)**6)**10
psi_1 = lambda z,s: np.exp(-z**2/(2*s**2))
psi_2 = lambda z,s,w: np.exp(-(z-w/2)**2/(2*s**2)) + np.exp(-(z+w/2)**2/(2*s**2))
psi_3 = lambda z,s,w,h_1: h_1*(np.exp(-(z-w)**2/(2*s**2)) + np.exp(-(z+w)**2/(2*s**2))) + psi_1(z,s)
psi_4 = lambda z,s,w,h_1: h_1*(np.exp(-(z-3*w/2)**2/(2*s**2)) + np.exp(-(z+3*w/2)**2/(2*s**2))) + psi_2(z,s,w)
psi_5 = lambda z,s,w,h_1,h_2: h_2*(np.exp(-(z-2*w)**2/(2*s**2)) + np.exp(-(z+2*w)**2/(2*s**2))) + psi_3(z,s,w,h_1)
psi_6 = lambda z,s,w,h_1,h_2: h_2*(np.exp(-(z-5*w/2)**2/(2*s**2)) + np.exp(-(z+5*w/2)**2/(2*s**2))) + psi_4(z,s,w,h_1)
funcs = (psi_0,psi_2,psi_3,psi_4,psi_5,psi_6)

init_guesses = {1:[1,1,1,0.02],
    2:[1,1,1,5],
    3:[1,1,1,5,0.9],
    4:[1,1,1,5,0.9],
    5:[1,1,1,5,0.8,0.6]}
