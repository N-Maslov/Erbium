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