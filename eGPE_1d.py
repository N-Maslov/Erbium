import numpy as np
import scipy.special as sc
from scipy.optimize import minimize
from scipy.fftpack import diff
import matplotlib.pyplot as plt
import time
import cProfile
from copy import copy, deepcopy

''' ~ ~ ~ SECTION 1: ENERGY CALCULATION ~ ~ ~  '''
# set parameters
D = 5.64e-3         # overall interaction strength (FIXED) omega = 150 gives 5.46e-4 for erbium, 0.0108 for Dy
e_dd = 1.40         # dipole interaction strength
a_ratio = 0.3       # trap aspect ratio, omega_z / omega_R 
N = 5.0e4           # number of particles 5.0e4 

# computational preferences
thetconstr = np.arctan(1/2**0.5) # max modulation
z_len = 1/a_ratio**0.5  # characteristic length of z-trap
RES = 2**10             # array length for integral and FFT, fastest w/ power of 2, must be EVEN

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

def f_x4m(N:int,L:float,in_vect:np.ndarray) -> np.ndarray:
    """Approximates input as periodic sampled input.
    Input vector has N entries between +-L/2.
    Returns Fourier transformed array centred on zero to best match FT definition,
    and corresponding k's."""
    ks = 2*np.pi*np.fft.fftfreq(N,L/N)
    x4m = L/N * np.exp(ks*L/2*1j) * np.fft.fft(in_vect)
    x4m = np.fft.fftshift(x4m)
    return x4m

def inv_f_x4m(N,k_range,in_vect):
    """Inverse fourier transform. Designed for compatibility with f_x4m."""
    inv_input = in_vect[::-1]
    x4m = f_x4m(N,k_range,np.concatenate(([0],inv_input[:-1])))
    return 1/(2*np.pi)*x4m

def particle_energy(psi_args,psi_0):
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
    psis = psis/N_corr**0.5
    psisq = psisq/N_corr
    F_psi_sq = f_x4m(RES,L,psisq)

    # get kinetic energies for each point
    KE_contribs = -0.5 * diff(psis,2,L)

    # get interaction contributions from each point
    Phis = np.real(inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq))

    # sum all energy terms
    return 0.25*(eta+1/eta)*(l**2+1/l**2) + step*(psisq @ (
        pref_QF*np.abs(psis/l)**3 + pref_inter/l**2*Phis + 1/2*a_ratio**2*zs**2
    ) + psis@KE_contribs)

def U_sig(ks,eta,l):
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

# set wavefunction
def psi_0(z,sigma,theta,period):
    """Must be of form psi_0(z,arg1, arg2, ...)
    Returns value of wavefunction given coordinate and psi_z parameters"""
    return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2)) *\
    (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))

def contrast(theta):
    '''Gets contrast from modulation strength parameter'''
    return (2**1.5 * np.sin(2*theta))/(3-np.cos(2*theta))

''' ~ ~ ~ SECTION 2: ENERGY MINIMISATION ~ ~ ~ '''
def gen_data(e_vals: np.ndarray, x_0: list,save=False,modtype=-1):
    '''Sweeps across e_dd, minimising the energy for each.
    Outputs lists of parameters and associated energies.'''

    # cannot be local variables as they are changed here and referenced in particle_energy
    global step, zs, ks, k_range, L, pref_inter, pref_QF,e_dd
    params = np.empty_like(e_vals,dtype=tuple)
    energies = np.zeros_like(e_vals,dtype=float)

    # theta bounds based on whether we're probing positive or negative modulation
    if modtype == -1:
        thetbnds = (-thetconstr,0)
        max_init_period = 10
    else:
        thetbnds = (0,thetconstr)
        max_init_period = 2

    for i,e_dd in enumerate(e_vals):
        # calculate interaction strengths for this e_dd
        pref_inter, pref_QF = get_coeff(D,e_dd,N)
    
        counter = 0
        while True:
            step, zs, ks, k_range, L = set_mesh(20*x_0[2])
            bnds = (0.9,None),(0.1,None),(10*step,None),thetbnds,(10*step,None)

            # prevent using modulation to vary wavefunction width
            if x_0[4] > max_init_period*x_0[2]:
                x_0[4] = max_init_period*x_0[2]/3
                #x_0[3] = 0

            # attempt to stimulate droplets
            if i % 10 == 0:
                x_0[3] = modtype*thetconstr

            res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')
            
            # upate starting point
            x_0 = deepcopy(res.x)*np.random.normal(1,0.05,(5))

            # Dynamic integration length set: break out only when grid is large enough to 
            # encompass wavefunction and fine enough to not miss small variations
            if 15*res.x[2] < L and L < 25*res.x[2]:
                break

            # force exit loop if infinite loop produced
            counter += 1
            if counter == 20:
                print(i, 'Unable to make correct gridsize: max iterations exceeded')
                break

        # append values of parameters and energy into relevant arrays
        params[i] = res.x
        energies[i] = res.fun

    # turn into single array for compatibility
    params = np.stack(params)

    # save data
    if save:
        np.savetxt('params.csv',params,delimiter=',')

    # eliminate pre-collapse noise from modulation width graph
    params[:,4] = np.where(np.abs(params[:,3])>0.01,params[:,4],np.zeros_like(e_vals))

    return params,energies

def gen_data_2d(e_min=1,e_max=2,e_num=20,a_min=0.02,a_max=0.5,a_num=10):
    '''Generates matrices containing values of each parameter and energies
    for a range of e_dd and aspect ratios specified in arguments.
    Returns an (n*m*6) matrix with them and arrays of e_dd and a_ratio'''

    # generate arrays for outputs
    global a_ratio, z_len
    xvalslist = np.linspace(e_min,e_max,e_num,endpoint=True)
    yvalslist = np.linspace(a_min,a_max,a_num,endpoint=True)
    outMat = np.zeros((a_num,e_num,6),dtype=float)

    # cycle through different aspect ratios
    for j, a_ratio in enumerate(yvalslist):
        z_len = 1/a_ratio**0.5

        # Initialise passes for all 4 combinations of forward and back through e_dd,
        # and positive and negative modulation.
        es = xvalslist,xvalslist,xvalslist[::-1],xvalslist[::-1]
        modtypes = -1,1,-1,1
        x_0s = ([2,2,2.5/a_ratio**0.8,0,z_len],[2,2,2.5/a_ratio**0.8,0,z_len],
        [10,2,1,-thetconstr,1.5],[10,2,1,thetconstr,1.5])

        # arrays to overwrite with values for minimum energy
        min_energies = 1000*np.ones_like(xvalslist,dtype=float)
        min_params   = np.zeros((e_num,5),dtype=float)

        # Out of 4 passes, points are chosen that have minimum energy.
        # This is to attempt to find global minima.
        for run in range(4):
            params, energies = gen_data(es[run],x_0s[run],modtype=modtypes[run])

            if run >= 2:
                # flip direction on arrays with wrong e_dd order
                params=np.flip(params,0)
                energies=np.flip(energies)
            replace = (energies<min_energies)
            for pos in range(e_num):
                # replace value in arrays if lower energy is found
                if replace[pos]:
                    min_energies[pos] = copy(energies[pos])
                    min_params[pos] = copy(params[pos])
            print(' ->',run)
            #plot_1d(axs,params,energies,xvalslist)
        outMat[j] = np.concatenate((min_params,np.array([min_energies]).T),axis=1)
        print(j)
        
    return outMat,xvalslist,yvalslist

def get_lifetimes(paramMat,k=1):
    '''Returns matrix of estimates of 3-body loss decay times based on first moment of density'''
    # Initialise output array
    lifeMat = np.empty(paramMat.shape[:2],dtype=float)
    for i, a_data in enumerate(paramMat):
        for j, params in enumerate(a_data):
            # initialise grid
            L_integ = 20*params[2]
            step = L_integ/RES
            z_vals = np.linspace(-L_integ/2,L_integ/2,RES)

            # normalise wavefunction
            psi_sq = psi_0(z_vals,*params[2:5])**2
            N_corr = np.sum(psi_sq)*step
            psi_sq = psi_sq/N_corr

            # calculate decay time and add to output matrix
            lifeMat[i,j] = 1/(k*np.sum(psi_sq**3))

        print('lifetimes:',i)
    return lifeMat

''' ~ ~ ~ SECTION 3: 1D DISPLAY ~ ~ ~ '''
def plot_1d(axs,params,energies,e_vals):
    '''Generates 6 plots as a function of e_dd:
    one for each parameter and one for energy.'''
    for x in [0,1,2]:
            axs[0,x].plot(e_vals,params[:,x],'.')

    axs[1,0].plot(e_vals,contrast(params[:,3]),'.')
    axs[1,1].plot(e_vals,params[:,4],'.')
    axs[1,2].plot(e_vals,energies,'.')

def decorate_1d(fig,axs,arat=a_ratio):
    """Adds axis labels and title to 1d plot of parameters and energy against e_dd"""
    for ax in np.reshape(axs,(6)):
        ax.set_xlabel('e_dd')
    axs[0,0].set_ylabel('obliquity eta')
    axs[0,1].set_ylabel('transverse width / (QHO width)')
    axs[0,2].set_ylabel('longitudinal width / (QHO width)')
    axs[1,0].set_ylabel('contrast')
    axs[1,1].set_ylabel('period / (QHO width)')
    axs[1,2].set_ylabel('minimised energy / (hbar omega)')
    fig.suptitle(f'Parameter minima as a function of e_dd (D={D},a_ratio={arat},N={N},res={RES})') 

''' ~ ~ ~ SECTION 4: RUNNING ~ ~ ~ '''
if __name__ == "__main__":
    fig, axs = plt.subplots(2,3) # for variables
    start_time = time.time()

    # generate data and store in arrays
    outMat,e_vals,a_vals = gen_data_2d(1.30,1.32,5,0.41,0.43,5)
    min_params = outMat[:,:,:5]
    min_energies = outMat[:,:,5]
    lifetimes = get_lifetimes(outMat)

    # save data to outputs folder
    names = ['outputs\\outputs_'+str(i)+'.csv' for i in range(6)]
    for i, name in enumerate(names):
        np.savetxt(name,outMat[:,:,i],delimiter=',')
    np.savetxt('outputs\\e_vals.csv',e_vals,delimiter=',')
    np.savetxt('outputs\\a_vals.csv',a_vals,delimiter=',')
    np.savetxt('outputs\\lifetimes.csv',lifetimes,delimiter=',')

    print('Total run time: ',time.time()-start_time)