import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from copy import deepcopy
from multiprocessing import Pool
import setup
import energy_calc
import param_calcs

def gen_data(D,N,n:int,e_vals:np.ndarray,a_ratio:float,x_0:list,save=False,plot=-1):
    '''Sweeps across e_dd, minimising the energy for each.
    Outputs lists of parameters and associated energies.'''

    # cannot be local variables as they are changed here and referenced in particle_energy
    params = np.empty_like(e_vals,dtype=tuple)
    energies = np.zeros_like(e_vals,dtype=float)

    func = setup.funcs[n-1]
    n_params = func.__code__.co_argcount+1

    bnds = [(None,None) if i <= 3 else (0,2) for i in range(n_params)]
    bnds[0], bnds[1] = (0.1,None),(0.01,None)

    for i,e_dd in enumerate(e_vals):
        # calculate interaction strengths for this e_dd
        pref_inter, pref_QF = setup.get_coeff(D,e_dd,N)

        counter = 0
        while True:
            if n == 1:
                mesh_args = setup.set_mesh(40*x_0[2])
                bnds[3] = (0,2)
            else:
                if n%2 == 0:
                    mesh_args = setup.set_mesh((n-1)*x_0[3]+40*x_0[2])
                else:
                    mesh_args = setup.set_mesh(n*x_0[3]+40*x_0[2])
                bnds[3] = (10*mesh_args[0],None)
            bnds[2] = (10*mesh_args[0],None)

            res = minimize(energy_calc.particle_energy,x_0,bounds=bnds,
                args=(func,mesh_args,setup.RES,e_dd,a_ratio,pref_inter,pref_QF),method='L-BFGS-B')
            
            # upate starting point
            x_0 = deepcopy(res.x)#*np.random.normal(1,0.05,(n_params))

            # Dynamic integration length set: break out only when grid is large enough to 
            # encompass wavefunction and fine enough to not miss small variations
            L_bnds = np.array([35*res.x[2],45*res.x[2]])
            if n%2 == 0:
                L_bnds += (n-1)*np.array([res.x[3],res.x[3]])
            elif n>1:
                L_bnds += n*np.array([res.x[3],res.x[3]])

            if L_bnds[0] < mesh_args[-1] and mesh_args[-1] < L_bnds[1]:
                break

            # force exit loop if infinite loop produced
            counter += 1
            if counter == 20:
                print(i, 'Unable to make correct gridsize: max iterations exceeded')
                break

        # append values of parameters and energy into relevant arrays
        params[i] = res.x
        energies[i] = res.fun

        if plot!=-1 and i%plot == 0:
            psisq = func(mesh_args[1],*params[i][2:])**2
            plt.plot(mesh_args[1],psisq/(np.sum(psisq)*mesh_args[0]))
            plt.title(str(params[i])+f'\ne_dd={e_dd}')
            plt.show()

    # turn into single array for compatibility
    params = np.stack(params)

    # save data
    if save:
        np.savetxt('params.csv',params,delimiter=',')

    return params,energies

def gen_data_2d(f=150.0,N=5.e4,e_min=1.25,e_max=1.5,e_num=20,a_min=0.02,a_max=0.5,a_num=10):
    '''Generates matrices containing values of each universal parameter and energies
    for a range of e_dd and aspect ratios specified in arguments.
    i: parameter
    j: aspect ratio'''
    D = setup.get_D(f)
    xvalslist = np.linspace(e_min,e_max,e_num,endpoint=True)
    yvalslist = np.linspace(a_min,a_max,a_num,endpoint=True)
    # numbers of droplets to try
    ns = [1,2,3,4,5]

    # generate output matrices
    outMat = np.zeros((10,a_num,e_num),dtype=float)
    outEnergies = np.zeros((6,a_num,e_num),dtype=float)
            
    # cycle through different aspect ratios
    for j, a_ratio in tqdm(enumerate(yvalslist),total=a_num):

        # cycle through different allowable numbers of droplets        
        if __name__ =='__main__':
            to_pass = [(D,N,n,xvalslist,a_ratio,setup.init_guesses[n]) for n in ns]
            with Pool() as p:
                n_tuples=p.starmap(gen_data,to_pass)
            params_arrays = [n_tuple[0] for n_tuple in n_tuples]
            energies_arrays = [n_tuple[1] for n_tuple in n_tuples]

        min_energies,min_params,min_ns=param_calcs.get_minima(params_arrays,energies_arrays,ns)

        for k,n in enumerate(min_ns):
            parameters = min_params[k]
            outMat[0,j,k] = param_calcs.get_contrast(n,parameters)
            outMat[2,j,k] = param_calcs.get_lifetime(n,parameters,param_calcs.loss_coeff(xvalslist[k],f,N))
            outMat[4,j,k] = parameters[0]
            outMat[5,j,k] = parameters[1]
            outMat[6,j,k] = parameters[2] if n>1 else 0 # droplet widths
            outMat[7,j,k] = parameters[3] if n>1 else 0 # droplet separations
            outMat[8,j,k] = parameters[4] if n>2 else 0 # 1st order decay
            outMat[9,j,k] = parameters[5] if n>4 else 0 # 2nd order decay

            #outEnergies[0,j,k] = energy_calc.energy_funcs[0]()
        outMat[1,j] = min_energies
        outMat[3,j] = min_ns

    settings = np.array((f,N,e_min,e_max,e_num,a_min,a_max,a_num))
    return outMat, settings


if __name__ == '__main__':
    mat, settings = gen_data_2d(150,5.e4,1.25,1.5,5,0.02,0.6,5)
    np.save('output\\outMat.npy',mat)
    np.save('output\\settings.npy',settings)