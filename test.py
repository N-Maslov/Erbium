from eGPE_1d import *
from scipy.special import zeta
def analytic_energy(psi_args):
    eta = psi_args[0]
    l = psi_args[1]
    sigma = psi_args[2]
    return 0*0.25*(eta+1/eta)*(l**2+1/l**2) +\
        0*1/(4*sigma**2) +\
        N*A/(np.sqrt(np.pi*2)*l**2*sigma)*(
            1+e_dd*((f(eta**0.25*l/sigma)+1-eta)/(1+eta))
    ) + 3*N*A*e_dd/L**3 * zeta(3) +\
        0*256/(15*np.pi)*(1+1.5*e_dd**2)*A**2.5/(l**3*sigma**1.5)*(2*N/(5*np.pi**0.5))**1.5 +\
        0*a_ratio**2/4 * sigma**2

def f(k):
    bit = np.arctanh(np.sqrt(1-k**2))/np.sqrt(1-k**2)
    return (1+2*k**2-3*k**2*bit)/(1-k**2)

print(particle_energy((1.1,0.9,z_len,0,z_len*0.17),psi_0))
print(analytic_energy((1.1,0.9,z_len)))

"""x_0 = (1,1,z_len)
bnds = ((0.9,None),(0.1,None),(0.1*z_len,75))

### Plots generation for QHO
e_vals = np.linspace(0,1.6,20)
min_etas = np.zeros_like(e_vals,dtype=float)
min_ls = deepcopy(min_etas)
min_sigmas = deepcopy(min_etas)
min_energies = deepcopy(min_etas)
for i,e_dd in enumerate(e_vals):
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term 
    res = minimize(analytic_energy,x_0,bounds=bnds)
    min_etas[i] = res.x[0]
    min_ls[i] = res.x[1]
    min_sigmas[i] = res.x[2]
    min_energies[i] = analytic_energy((res.x[0],res.x[1],res.x[2]))
    print(i)


### PLOTTING ###
axs[0,0].plot(e_vals,min_etas)
axs[0,1].plot(e_vals,min_ls)
axs[1,0].plot(e_vals,min_sigmas)
axs[1,1].plot(e_vals,min_energies)
plt.show()"""