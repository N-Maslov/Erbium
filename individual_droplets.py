from typing import Callable
import numpy as np
import scipy.special as sc
from scipy.optimize import minimize
from scipy.fftpack import diff
import matplotlib.pyplot as plt
import cProfile
from copy import deepcopy, copy
import matplotlib.animation as animation
import warnings
warnings.filterwarnings('error')

''' ~ ~ ~ SECTION 1: ENERGY CALCULATION ~ ~ ~  '''
# set parameters
D = 5.46e-3         # overall interaction strength (FIXED) omega = 150 gives 5.46e-3 for erbium, 0.0108 for Dy
e_dd = 1.40         # dipole interaction strength
a_ratio = 0.3       # trap aspect ratio, omega_z / omega_R 
N = 5.0e4           # number of particles 5.0e4 

# computational preferences
z_len = 1/a_ratio**0.5  # characteristic length of z-trap
RES = 2**12             # array length for integral and FFT, fastest w/ power of 2, must be EVEN

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

def particle_energy(psi_args:tuple,psi_0:Callable) -> float:
    """Calculate dimensionless per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take numpy vector inputs!
    modify with conjugate, abs to allow complex psi.
    Returns energy."""

    # extract wavefunction parameters from supplied arguments
    eta = psi_args[0]
    l = psi_args[1]
    psi_0_args = psi_args[2:]

    # calculate and normalise wavefunction
    psis = psi_0(zs,*psi_0_args)
    psisq = psis**2
    N_corr = np.sum(psisq)*step
    try:
        psis = psis/N_corr**0.5
        psisq = psisq/N_corr
    except RuntimeWarning:
        print(N_corr)
    F_psi_sq = f_x4m(RES,L,psisq)

    # get kinetic energies for each point
    KE_contribs = -0.5 * diff(psis,2,L)

    # get interaction contributions from each point
    Phis = np.real(inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq))

    # sum all energy terms
    return 0.25*(eta+1/eta)/l**2 + 0.25*l**2*((151/227)**2*eta+1/eta) + step*(psisq @ (
        pref_QF*np.abs(psis/l)**3 + pref_inter/l**2*Phis + 1/2*a_ratio**2*zs**2
    ) + psis@KE_contribs)

def U_sig(ks:np.ndarray,eta:float,l:float) -> np.ndarray:
    """Calculate approximation function for 2D fourier transform
    (intermediate calculation for particle_energy)"""
    Q_sqs = 1/2*eta**0.5*(ks*l)**2
    Q_sqs = np.where(Q_sqs<703,Q_sqs,703*np.ones_like(ks,dtype=float))
    # low value limit: ks is 0 at RES/2
    Q_sqs[int(RES/2)] = 1.e-18
    # normal calculation
    numerat = np.where(Q_sqs<703,
        3*(Q_sqs*np.exp(Q_sqs)*sc.expi(-Q_sqs)+1),3/Q_sqs)
    # high value limit automatically zero now
    return 1+e_dd * (numerat/(1+eta)-1)

# set wavefunctions
psi_1 = lambda z,s: np.exp(-z**2/(2*s**2))
psi_2 = lambda z,s,w: np.exp(-(z-w/2)**2/(2*s**2)) + np.exp(-(z+w/2)**2/(2*s**2))
psi_3 = lambda z,s,w,h: np.exp(-h)*(np.exp(-(z-w)**2/(2*s**2)) + np.exp(-(z+w)**2/(2*s**2))) + psi_1(z,s)
psi_4 = lambda z,s,w,h: np.exp(-h)*(np.exp(-(z-3*w/2)**2/(2*s**2)) + np.exp(-(z+3*w/2)**2/(2*s**2))) + psi_2(z,s,w)
psi_5 = lambda z,s,w,h: np.exp(-2*h)*(np.exp(-(z-2*w)**2/(2*s**2)) + np.exp(-(z+2*w)**2/(2*s**2))) + psi_3(z,s,w,h)

funcs = (psi_1,psi_2,psi_3,psi_4,psi_5)

### minimiser ###
def gen_data(n,e_vals: np.ndarray, x_0: list,save=False,plot=-1):
    '''Sweeps across e_dd, minimising the energy for each.
    Outputs lists of parameters and associated energies.'''

    # cannot be local variables as they are changed here and referenced in particle_energy
    global step, zs, ks, k_range, L, pref_inter, pref_QF,e_dd
    params = np.empty_like(e_vals,dtype=tuple)
    energies = np.zeros_like(e_vals,dtype=float)
    func = funcs[n-1]
    bnds = [(0.1,None),(0.01,None),()]
    if n == 1:
        n_params = 3
    elif n ==2:
        n_params = 4
    else:
        n_params = 5
    bnds = [() for _ in range(n_params)]
    bnds[0], bnds[1] = (0.1,None),(0.01,None)
    if n_params == 5:
        bnds[4] = (0,None)

    for i,e_dd in enumerate(e_vals):
        # calculate interaction strengths for this e_dd
        pref_inter, pref_QF = get_coeff(D,e_dd,N)

        counter = 0
        while True:
            if n == 1:
                step, zs, ks, k_range, L = set_mesh(40*x_0[2])
            else:
                if n%2 == 0:
                    step, zs, ks, k_range, L = set_mesh((n-1)*x_0[3]+40*x_0[2])
                else:
                    step, zs, ks, k_range, L = set_mesh(n*x_0[3]+40*x_0[2])
                bnds[3] = (10*step,None)
            bnds[2] = (10*step,None)

            res = minimize(particle_energy,x_0,bounds=bnds,args=(func),method='L-BFGS-B')
            
            # upate starting point
            x_0 = deepcopy(res.x)*np.random.normal(1,0.05,(n_params))

            # Dynamic integration length set: break out only when grid is large enough to 
            # encompass wavefunction and fine enough to not miss small variations
            L_bnds = np.array([35*res.x[2],45*res.x[2]])
            if n%2 == 0:
                L_bnds += (n-1)*np.array([res.x[3],res.x[3]])
            elif n>1:
                L_bnds += n*np.array([res.x[3],res.x[3]])

            if L_bnds[0] < L and L < L_bnds[1]:
                break

            # force exit loop if infinite loop produced
            counter += 1
            if counter == 20:
                print(i, 'Unable to make correct gridsize: max iterations exceeded')
                break

        # append values of parameters and energy into relevant arrays
        params[i] = res.x
        energies[i] = res.fun

        print(i)
        if plot!=-1 and i%plot == 0:
            psisq = func(zs,*params[i][2:])**2
            plt.plot(zs,psisq/(np.sum(psisq)*step))
            plt.title(str(params[i])+f'\ne_dd={e_dd}')
            plt.show()

    # turn into single array for compatibility
    params = np.stack(params)

    # save data
    if save:
        np.savetxt('params.csv',params,delimiter=',')

    return params,energies

def plot_1d(n,fig,axs,params,energies,e_vals,arat):
    '''Generates 6 plots as a function of e_dd:
    one for each parameter and one for energy.'''
    for x in [0,1,2]:
            axs[0,x].plot(e_vals,params[:,x],label=str(n))
    axs[0,0].set_ylabel('obliquity eta')
    axs[0,1].set_ylabel('transverse width / (QHO width)')
    axs[0,2].set_ylabel('droplet width / (QHO width)')

    if n>1:
        axs[1,0].plot(e_vals,params[:,3],label=str(n))
        axs[1,0].set_ylabel('droplet separation / (QHO width)')
        if n>2:
            axs[1,1].plot(e_vals,params[:,4],label=str(n))
            axs[1,1].set_ylabel('droplet height decay')

    axs[1,2].plot(e_vals,energies,label=str(n))
    axs[1,2].set_ylabel('minimised energy / (hbar omega)')

    for ax in np.reshape(axs,(6)):
        ax.set_xlabel('e_dd')
        ax.legend()

    fig.suptitle(f'Parameter minima as a function of e_dd (D={D},a_ratio={arat},N={N},res={RES})') 

def animate(i,ax1,ns,params,xvalslist):
    ax1.clear()
    ax1.grid()
    psisq = funcs[ns[i]-1](zs,*params[i][2:])**2
    ax1.plot(zs,psisq/(np.sum(psisq)*step))
    ax1.text(0.05,0.8,
        'a_s = '+"{:.3f}".format(65.5/xvalslist[i]),
    fontsize = 'large',transform=ax1.transAxes)
    
D = get_D(227)
a_ratio = 0.1388
if __name__ == '__main__':
    # generate data
    e_vals=np.linspace(1.25,1.5,200)[::1]

    params1,energies1 = gen_data(1,e_vals,[1,1,1])
    params2,energies2 = gen_data(2,e_vals,[1,1,1,5])
    params3,energies3 = gen_data(3,e_vals,[1,1,1,5,0.9])
    params4,energies4 = gen_data(4,e_vals,[1,1,1,5,0.9],plot=-1)
    params5,energies5 = gen_data(5,e_vals,[1,1,1,5,0.9],plot=-1)

    # find where energy is minimum
    params = params1,params2,params3,params4,params5
    energies = energies1,energies2,energies3,energies4,energies5

    min_energies = 1000*np.ones_like(e_vals,dtype=float)
    min_params = np.empty_like(e_vals,dtype=object)
    min_ns = np.empty_like(e_vals,dtype=int)

    for run in range(5):
        n = run+1
        replace = (energies[run] < min_energies)
        for pos in range(len(e_vals)):
            if replace[pos]:
                min_energies[pos] = copy(energies[run][pos])
                min_params[pos] = copy(params[run][pos])
                min_ns[pos] = n

    params=np.stack(params1),np.stack(params2),np.stack(params3),np.stack(params4),np.stack(params5)
    for i in range(5):
        np.savetxt('params'+str(i)+'.csv',params[i],delimiter=',')
        np.savetxt('energies'+str(i)+'.csv',energies[i],delimiter=',')

    # animate evolution
    fig1,ax1 = plt.subplots()

    ani = animation.FuncAnimation(fig1, animate, range(len(e_vals)),
        interval=100,fargs=(ax1,min_ns,min_params,e_vals))
    plt.show()
    ani.save('psi_evolution.gif','pillow')

    fig,axs=plt.subplots(2,3)
    plot_1d(5,fig,axs,params5,energies5,65.5/e_vals,a_ratio)
    plot_1d(4,fig,axs,params4,energies4,65.5/e_vals,a_ratio)
    plot_1d(3,fig,axs,params3,energies3,65.5/e_vals,a_ratio)
    plot_1d(2,fig,axs,params2,energies2,65.5/e_vals,a_ratio)
    plot_1d(1,fig,axs,params1,energies1,65.5/e_vals,a_ratio)
 
    plt.show()