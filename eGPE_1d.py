import numpy as np
import scipy.special as sc
import ftransformer as ft
from scipy.optimize import minimize
from scipy.fftpack import diff
import warnings
import matplotlib.pyplot as plt
import cProfile
from copy import deepcopy
warnings.filterwarnings('error')

# set parameters
D = 5.64e-3         # overall interaction strength (FIXED) omega = 150 gives 5.46e-4 for erbium, 0.0108 for Dy
e_dd = 1.40         # dipole interaction strength
a_ratio = 0.3      # trap aspect ratio, omega_z / omega_R 
N = 0.5e5            # number of particles 5.0e4 

# computational preferences
z_len = 1/a_ratio**0.5  # characteristic length of z-trap
RES = 2**10             # array length for integral and FFT, fastest w/ power of 2, must be EVEN ##################################

def set_mesh(L):
    '''gets computational constants (step, zs, ks, k_range, l) for given grid size L'''
    step = L / RES # z increment
    zs = np.linspace(-L/2,L/2,RES,endpoint=False)
    ks = np.fft.fftshift(2*np.pi*np.fft.fftfreq(RES,step))
    k_range = ks[-1]-2*ks[0]+ks[1]
    return step, zs, ks, k_range, L

def get_coeff(D,e_dd,N):
    '''gets coefficients for interaction and QF terms from parameters supplied'''
    A = D/e_dd # dimensionless scattering length
    pref_inter = A*N # prefactor for interaction term
    pref_QF = 512/(75*np.pi) * A**2.5 * N**1.5 * (1+1.5*e_dd**2) # prefactor for QF term
    return pref_inter, pref_QF

def particle_energy(psi_args,psi_0):
    """Calculate per-particle energy given
     - longitudinal functional form psi_0(z,*args)
     - arguments for psi, the first two of which must be anisotropy eta
     and mean width l.
    Does not take numpy vector inputs!
    modify with conjugate, abs to allow complex psi."""

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
    F_psi_sq = ft.f_x4m(RES,L,psisq)

    # get kinetic energies for each point
    KE_contribs = -0.5 * diff(psis,2,L)

    # get interaction contributions from each point
    Phis = np.real(ft.inv_f_x4m(RES,k_range,U_sig(ks,eta,l)*F_psi_sq))

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
    """Must be of form psi_0(z,arg1, arg2, ...)"""
    return (1/(np.sqrt(np.pi)*sigma))**0.5 * np.exp(-z**2/(2*sigma**2)) *\
    (np.cos(theta) + 2**0.5*np.sin(theta)*np.cos(2*np.pi*z/period))

# get maximum theta
thetconstr = np.arctan(1/2**0.5)

# get contrast from modulation strength parameter
contrast = lambda theta: (2**1.5 * np.sin(2*theta))/(3-np.cos(2*theta))


def gen_data(e_vals: np.ndarray, x_0: list, plotfreq=-1,save=False,modtype=-1):
    '''Outputs lists of parameters, grid resolutions at each point, and associated energies.'''

    # cannot be local variables as they are changed here and referenced in particle_energy
    global step, zs, ks, k_range, L, pref_inter, pref_QF,e_dd
    params = np.empty_like(e_vals,dtype=tuple)
    len_res = np.zeros_like(e_vals,dtype=float)
    energies = np.zeros_like(e_vals,dtype=float)

    # theta bounds based on whether we're probing positive or negative modulation
    if modtype == -1:
        thetbnds = (-thetconstr,0)
    else:
        thetbnds = (0,thetconstr)

    for i,e_dd in enumerate(e_vals):
        # calculate interaction strengths for this e_dd
        pref_inter, pref_QF = get_coeff(D,e_dd,N)
    
        while True:
            step, zs, ks, k_range, L = set_mesh(20*x_0[2])
            bnds = (0.9,None),(0.1,None),(10*step,None),thetbnds,(10*step,None)

            # prevent using modulation to vary wavefunction width
            if x_0[4] > 2*x_0[2]:
                x_0[4] = x_0[2] / 2
                #x_0[3] = 0

            res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')
            
            # upate starting point
            x_0 = deepcopy(res.x)#*np.random.normal(1,0.1,(5))

            # break out only when grid is large enough to encompass wavefunction
            # and fine enough to not miss small variations
            if 15*res.x[2] < L and L < 25*res.x[2]:
                break

        params[i] = res.x
        len_res[i] = 10*step/z_len
        energies[i] = particle_energy(res.x,psi_0)

        # plot wavefunctions
        if plotfreq != -1:
            if i % plotfreq == 0:
                psisq = psi_0(zs,*res.x[2:])**2
                ax2.plot(zs,psisq/(np.sum(psisq)*step))
        print(i)

    # turn into single array
    params = np.stack(params)

    # save data
    if save:
        np.savetxt('params.csv',params,delimiter=',')

    # eliminate pre-collapse noise from modulation width graph
    params[:,4] = np.where(np.abs(params[:,3])>0.01,params[:,4],np.zeros_like(e_vals))

    return params,energies,len_res,e_vals

def plot_1d(axs,params,energies,len_res,e_vals):
    for x in [0,1,2]:
            axs[0,x].plot(e_vals,params[:,x],'.')

    #axs[0,2].plot(e_vals,len_res)
    #axs[1,0].plot(e_vals,params[:,3],'.')
    axs[1,0].plot(e_vals,contrast(params[:,3]),'.')
    axs[1,1].plot(e_vals,params[:,4],'.')
    #axs[1,1].plot(e_vals,len_res)
    axs[1,2].plot(e_vals,energies,'.')

### PLOTTING ###
if __name__ == "__main__":
    fig, axs = plt.subplots(2,3) # for variables
    fig2,ax2 = plt.subplots()    # for wavefunction

    e_vals = np.linspace(1.3,1.36,100)
    es = e_vals,e_vals,e_vals[::-1],e_vals[::-1]
    modtypes = -1,1,-1,1
    x_0s = ([2,2,10*z_len,0,z_len],[2,2,10*z_len,0,z_len],
     [10,2,z_len,-thetconstr,0.9*z_len],[10,2,z_len,thetconstr,0.9*z_len])

    # run twice and plot both sets of data, going from min and max e_dd.
    for run in range(4):
        outs = gen_data(es[run],x_0s[run],modtype=modtypes[run])
        plot_1d(axs,*outs)

    # decorate with labels and all
    for ax in np.reshape(axs,(6)):
        ax.set_xlabel('e_dd')
    axs[0,0].set_ylabel('obliquity eta')
    axs[0,1].set_ylabel('transverse width / (QHO width)')
    axs[0,2].set_ylabel('longitudinal width / (QHO width)')
    axs[1,0].set_ylabel('contrast')
    axs[1,1].set_ylabel('period / (QHO width)')
    axs[1,2].set_ylabel('minimised energy / (hbar omega)')

    fig.suptitle(f'Parameter minima as a function of e_dd (D={D},a_ratio={a_ratio},N={N},res={RES})')  

    # victory
    plt.show()


#2D PLOT
######################################################################
"""SIZE = 75
yvalslist = np.linspace(0.02,1,SIZE,endpoint=True)
xvalslist = np.linspace(1,2,SIZE,endpoint=True)

outMat = np.zeros((SIZE,SIZE,6),dtype=float)
for i, a_ratio in enumerate(yvalslist):
    z_len = 1/a_ratio**0.5  # characteristic length of z-trap
    step, zs, ks, k_range, L = set_mesh(50*z_len)
    x_0 = 1,1,z_len,0,0.2*z_len
    bnds = (0.9,None),(0.1,None),(10*step,L),(-thetconstr,thetconstr),(10*step,L)
    transitioned = False

    for j, e_dd in enumerate(xvalslist):
        pref_inter, pref_QF = get_coeff(D,e_dd,N)
        res = minimize(particle_energy,x_0,bounds=bnds,args=(psi_0),method='L-BFGS-B')

        outMat[i][j][:-1] = res.x
        outMat[i][j][2] /= z_len
        outMat[i][j][3] = contrast(outMat[i][j][3])
        outMat[i][j][4] /= z_len
        outMat[i][j][-1] = particle_energy(res.x,psi_0)
        if not transitioned:
            if res.x[2]<x_0[2]/2:
                transitioned = True
                step, zs, ks, k_range, L = set_mesh(2.5*z_len)
                bnds = (0.9,None),(0.1,None),(10*step,None),(-thetconstr,0.0000),(10*step,None)

        x_0 = res.x*np.random.normal(1,0.01,(5))
        #print(j)
    print('NEW i = ',i)


names = ['outputs_'+str(i)+'.csv' for i in range(6)]
for i, name in enumerate(names):
    np.savetxt(name,outMat[:,:,i],delimiter=',')
vals = np.savetxt('vals.csv',[xvalslist,yvalslist],delimiter=',')
#"""