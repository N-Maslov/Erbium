from eGPE_1d import *

### TESTING ###
# To copy and paste in main pgm if necessary. All tests here have been passed.

# PLOT TEST:
ls = [6e-4]
etas = np.linspace(0.5,1.5)
energies = energies_mat(etas,ls)
import matplotlib.pyplot as plt
plt.plot(etas,energies[:,0])
plt.show()


# ANALYTIC TEST:
def analytic_energy(eta,l):
    gam_sig = 2/(5*np.pi*1.5*l**3)
    g_QF = gam_QF*gam_sig
    return 1/2*n*(g_s/(2*np.pi*l**2) + g_dd/(2*np.pi*l**2)*(2-eta)/(1+eta)) + 2/5*g_QF*n**1.5 +\
        hbar**2/(4*m*l**2)*(eta+1/eta) + m*l**2/4*(omegas[0]**2/eta+omegas[1]**2*eta)
energy_func1 = lambda x: analytic_energy(x[0],x[1])*1.e26
res = minimize(energy_func1,(2.,0.01),bounds=bnds)
print(res.x)

# Analytic test for cosine oscillation
from eGPE_1d import *
def analytic_energy(eta,l,theta,L_psi):
    val = hbar**2/(4*m*l**2)*(eta+1/eta) + m*l**2/4*(omegas[0]**2/eta+omegas[1]**2*eta)
    gam_sig = 2/(5*np.pi*1.5*l**3)
    g_QF = gam_QF*gam_sig
    Lambda = 1/32*(90*np.cos(theta)-55*np.cos(3*theta)-3*np.cos(5*theta))
    return val + hbar**2*np.sin(theta)**2/(2*m*L_psi**2) + 2/5*g_QF*Lambda*n**1.5 + n/2*(
        U_sig(np.array([0]),eta,l)[0] + np.sin(2*theta)**2*U_sig(np.array([2*np.pi/L_psi]),eta,l)[0]\
            +1/2*np.sin(theta)**4*U_sig(np.array([4*np.pi/L_psi]),eta,l)[0]
        )   



print(analytic_energy(1.08,5.63e-4,0.1,0.01))
energy_func1 = lambda x: analytic_energy(x[0],x[1],x[2],x[3])*1.e30
res = minimize(energy_func1,x_0,bounds=bnds)
print(res.x)
print(res.message)