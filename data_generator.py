'''
### DATA GENERATION ###
Runs are initiated by the user from the __main__ section of this program.
Contains function to generate 2D matrices of data, which repeatedly calls a function that does
it for a single aspect ratio, sweeping across e_dd. Results are saved in the output folder.
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import setup
import energy_calc
import param_calcs

def gen_data(D:float,N:float,n:int,e_vals:np.ndarray,a_ratio:float,x_0:list,save=False,plot=-1)->tuple:
    '''Sweeps across e_dd, minimising the energy for each value. Called by gen_data_2D but can also be
    used standalone. Outputs lists of parameters and associated energies for a given droplet number.
    Inputs:
    - D: dimensionless interaction strength a_dd / sqrt(hbar / m omega)
    - N: number of atoms in trap
    - n: number of droplets (fixed)
    - e_vals: array of values of e_dd
    - a_ratio: aspect ratio of trap; omega_z / omega_r. < 1 for elongated trap.
    - x_0 initial guess of parameters
    - save: if enabled, parameters will be saved as a .csv file
    - plot: will generate a plot of the wavefunction every (plot) calculations. -1 (default) to disable.
    Returns:
    - params: matrix of parameters. params[i,j] is jth parameter for ith value of e_dd.
    - energies: the corresponding energies. energies[i] is energy for ith value of e_dd.'''

    # create empty arrays to populate with energies and parameters
    params = np.empty_like(e_vals,dtype=tuple)
    energies = np.zeros_like(e_vals,dtype=float)

    # select function corresponding to chosen number of droplets
    func = setup.funcs[n-1]
    # get number of parameters required for this wavefunction (-1 because one of the arguments is the
    # z values, and +2 because there are two more parameters from the transverse wavefunction.)
    n_params = func.__code__.co_argcount+1

    # select bounds for minimisation. (0,2) for relative decay parameters.
    bnds = [(None,None) if i <= 3 else (0.2,2) for i in range(n_params)]
    bnds[0], bnds[1] = (0.1,None),(0.01,None) # prevent ZeroDivisionError for transverse part


    # sweep through values of e_dd (default: from smallest to largest, gives better behaviour)
    for i,e_dd in enumerate(e_vals):
        # calculate interaction strengths for this e_dd
        pref_inter, pref_QF = setup.get_coeff(D,e_dd,N)

        counter = 0
        # loop while dynamically setting integration range, until it is large enough to include
        # entire wavefunction but small enough to see narrow droplets.
        while True:
            # set integration length and corresponding bounds depending on number of droplets
            if n == 1:
                mesh_args = setup.set_mesh(40*x_0[2])
            else:
                if n%2 == 0:
                    mesh_args = setup.set_mesh((n-1)*x_0[3]+40*x_0[2])
                else:
                    mesh_args = setup.set_mesh(n*x_0[3]+40*x_0[2])
                bnds[3] = (10*mesh_args[0],None)
            bnds[2] = (10*mesh_args[0],None)

            # actually do the energy minimisation on this grid
            res = minimize(energy_calc.particle_energy,x_0,bounds=bnds,
                args=(func,mesh_args,setup.RES,e_dd,a_ratio,pref_inter,pref_QF),method='L-BFGS-B')
            
            # upate starting point for next run. Optional random modulation on top.
            x_0 = deepcopy(res.x)*np.random.normal(1,0.05,(n_params))

            # Break out when result of right size compared to integration length
            L_bnds = np.array([35*res.x[2],45*res.x[2]])
            if n%2 == 0:
                L_bnds += (n-1)*np.array([res.x[3],res.x[3]])
            elif n>1:
                L_bnds += n*np.array([res.x[3],res.x[3]])

            if L_bnds[0] < mesh_args[-1] and mesh_args[-1] < L_bnds[1]:
                break

            # force exit loop if infinite loop produced (just in case)
            counter += 1
            if counter == 20:
                print(i, 'Unable to make correct gridsize: max iterations exceeded')
                break

        # append values of parameters and energy into relevant arrays
        params[i] = res.x
        energies[i] = res.fun

        # plot longitudinal wavefunction if option is selected
        if plot!=-1 and i%plot == 0:
            psisq = func(mesh_args[1],*params[i][2:])**2
            plt.plot(mesh_args[1],psisq/(np.sum(psisq)*mesh_args[0]))
            plt.title(str(params[i])+f'\ne_dd={e_dd}')
            plt.show()

    # turn into single array for compatibility (previously array of arrays)
    params = np.stack(params)

    # save data
    if save:
        np.savetxt('output\\params.csv',params,delimiter=',')

    return params,energies

def gen_data_2d(f=150.0,N=5.e4,e_min=1.25,e_max=1.5,e_num=20,a_min=0.02,a_max=0.5,a_num=10)->tuple:
    '''Minimises energy across a 2D grid with variable e_dd and aspect ratio.
    Inputs:
    - f: frequency of transverse trap (Hz)
    - N: total number of atoms
    - e_min, e_max: minimum and maximum values of e_dd
    - e_num: total number of values of e_dd to probe
    - a_min, a_max: minimum and maximum values of the trap aspect ratio
    - a_num: total number of aspect ratio values to probe
    Returns:
    - outMat: matrix containing all the parameters, contrasts, energies, lifetimes, particle numbers.
        outMat[i,j,k] is ith parameter, jth aspect ratio, kth value of e_dd.
    - outEnergies: same format as outMat but containing the individual energy contributions
        (transverse kinetic, potential, quantum fluctuations, etc).
    - settings: array used to reconstruct the conditions of the run when data is loaded.
        Stores the input arguments passed to gen_data_2d on this run.'''
    
    # work out dimensionless interaction strength from frequency supplied
    D = setup.get_D(f)

    # generate arrays of e_dd and aspect ratio values to run calculations for
    xvalslist = np.linspace(e_min,e_max,e_num,endpoint=True)
    yvalslist = np.linspace(a_min,a_max,a_num,endpoint=True)

    # specify numbers of droplets to try
    ns = [1,2,3,4,5]

    # generate output matrices
    outMat = np.zeros((10,a_num,e_num),dtype=float)
    outEnergies = np.zeros((6,a_num,e_num),dtype=float)
            
    # cycle through different aspect ratios (and create progress bar)
    for j, a_ratio in tqdm(enumerate(yvalslist),total=a_num):

        # create a separate process for each droplet and minimise energy for it       
        if __name__ =='__main__':
            to_pass = [(D,N,n,xvalslist,a_ratio,setup.init_guesses[n]) for n in ns]
            with Pool(min(5,cpu_count())) as p:
                n_tuples=p.starmap(gen_data,to_pass)
            params_arrays = [n_tuple[0] for n_tuple in n_tuples]
            energies_arrays = [n_tuple[1] for n_tuple in n_tuples]

        # find overall energy minima by comparing results for all numbers of droplets
        min_energies,min_params,min_ns=param_calcs.get_minima(params_arrays,energies_arrays,ns)

        # output results to the main matrix
        outMat[1,j] = min_energies
        outMat[3,j] = min_ns

        # cycle through points along e_dd (action cannot be done elementwise)
        for k,n in enumerate(min_ns):
            parameters = min_params[k]

            # generate normalised minimum energy wavefunction for calcualtions of lifetime etc
            L_set = 40*parameters[2]
            if n%2 == 0: # even droplets
                L_set += (n-1)*parameters[3]
            elif n>1: # odd droplets
                L_set += n*parameters[3]

            func = setup.funcs[n-1]
            step, zs, ks, k_range, L = setup.set_mesh(L_set)

            psis = func(zs,*parameters[2:])
            psisq = psis**2
            N_corr = np.sum(psisq)*step
            psis = psisq/N_corr**0.5
            psisq = psisq/N_corr

            # get parameters for these calculations
            e_dd = xvalslist[k]
            pref_inter, pref_QF = setup.get_coeff(D,e_dd,N)
            eta, l = parameters[:2]

            # put results into main output matrix
            outMat[0,j,k] = param_calcs.get_contrast(psisq,setup.RES,n)
            outMat[2,j,k] = param_calcs.get_lifetime(psisq,step,l,e_dd,f,N)
            outMat[4,j,k] = parameters[0] # transverse obliquity
            outMat[5,j,k] = parameters[1] # transverse length
            outMat[6,j,k] = parameters[2] # droplet widths
            outMat[7,j,k] = parameters[3] if n>1 else 0 # droplet separations
            outMat[8,j,k] = parameters[4] if n>2 else 0 # 1st order decay
            outMat[9,j,k] = parameters[5] if n>4 else 0 # 2nd order decay

            # populate energies matrix with individual energy contributions
            outEnergies[0,j,k] = energy_calc.energy_funcs[0](
                psisq,step,ks,k_range,L,eta,l,pref_inter,e_dd,setup.RES)
            outEnergies[1,j,k] = energy_calc.energy_funcs[1](psis,psisq,step,l,pref_QF)
            outEnergies[2,j,k] = energy_calc.energy_funcs[2](eta,l)
            outEnergies[3,j,k] = energy_calc.energy_funcs[3](eta,l)
            outEnergies[4,j,k] = energy_calc.energy_funcs[4](psis,step,L)
            outEnergies[5,j,k] = energy_calc.energy_funcs[5](psisq,step,zs,a_ratio)

    # save run arguments for later reference
    settings = np.array((f,N,e_min,e_max,e_num,a_min,a_max,a_num))
    return outMat, outEnergies, settings


if __name__ == '__main__':
    mat, energies, settings = gen_data_2d(150,5.e4,1.25,1.5,10,0.02,0.6,5)
    np.save('output\\outMat.npy',mat)
    np.save('output\\outEnergies.npy',energies)
    np.save('output\\settings.npy',settings)