# Erbium #
## Programs for constructing phase diagrams of ultracold dipolar gases by numerically minimising the extended Gross-Pitaevskii Hamiltonian ##
The program is based on the approximation used in [P Blair, Blakie, et al 2020 Commun. Theor. Phys. 72 085501] of a constant Gaussian ansatz
for the longitudinal ansatz, reducing the 3d energy minimisation problem to a 1d one and allowing fast construction of a phase diagram
(runtime of a few hours on a decent PC for a roughly 200x200 grid). By default, the program uses a longitudinal ansatz of a fixed number of
Gaussian-droplets. For a given number of atoms, trap frequency, scattering length and aspect ratio, minimisation is done for every number of droplets
from one to an upper limit (default 5), and the parameters with the lowest energy are selected.

The final result is a matrix storing the result of the contrast, energy, decay time, number of droplets, and all wavefunction parameters
across a grid of varying e_dd and trap aspect ratio, for a fixed radial trap frequecy and number of atoms. It also outputs an energies matrix,
storing terms for each individual contribution to the Hamiltonian at every point.

Required PyPi packages: numpy, scipy, matplotlib, tqdm

The repository is split across five programs, each of which has a more detailed description in its first few lines.
Roughly speaking, functions are split across them as follows:
* data_generator: accessed by the user at the start of a run. Contains functions to generate data for a phase diagram
* data_plotter: accessed by the user after data generation is complete. Contains functions for visualisation of results.
* setup: can be accessed to change configuration settings, e.g. forms of the trial wavefunctions, the resolution, and the type of atom being used.
* energy_calc: contains functions to calculate the energy given some functional form of a wavefunction and set of parameters.
* param_calcs: contains functions to calculate lifetime and other important quantities after minimisation
* old: contains legacy files, most notably eGPE1d which uses the old modulated Gaussian ansatz. The animation and matplot functions
  are legacy versions of the functions in data_plotter designed for the old file format. (I can modify the old format if required).

A typical run will be conducted as follows:
In the data_generator.py program, the select radial trap frequency, number of atoms, and start, end and number of points for e_dd
and aspect ratio respectively. The results are saved in the outputs folder.
```
if __name__ == '__main__':
    mat, energies, settings = gen_data_2d(150,5.e4,1.25,1.5,10,0.02,0.6,5)
    np.save('output\\outMat.npy',mat)
    np.save('output\\outEnergies.npy',energies)
    np.save('output\\settings.npy',settings)
```

To plot the data, use the data_plotter program. Start by loading the files after the data generation has been done:
```
if __name__ == '__main__':
    mat = np.load('output\\outMat.npy')
    energies = np.load('output\\outEnergies.npy')
    settings = np.load('output\\settings.npy')
```
Then, there are several options. From the loaded files, to plot a 2D graph of, say, the decay time, which is corresponds to the 2nd term
(indexed from 0) in mat, the call would look as follows:
`plot_2d(mat,settings,2,False)`
If, instead, we want to plot a single contribution to the energy, for instance the quantum fluctuations which is the 1st term, the energies
option is instead set to True:
`plot_2d(mat,settings,1,True)`
If we would like to generate a one-dimensional constant-aspect ratio plot of, say, the 14th aspect ratio slice, the call would be as follows:
`plot_1d(mat,settings,14)`
Finally, if we would like to generate an animation of the evolution with e_dd across the same slice and save it as a gif, the call would be
`ani_generator(mat,settings,14,save=True,fps = 20)

Documentation on every individual function is found within their docstrings.
If anything isn't working correctly, please email me :) 
