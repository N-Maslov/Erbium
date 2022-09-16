import numpy as np
import matplotlib.pyplot as plt

def plot_1d(matrix:np.ndarray,settings:np.ndarray,slice_index:int):
    '''Generates 6 plots as a function of e_dd:
    one for each parameter and one for energy.'''
    fig, axs = plt.subplots(2,5)

    f,N,e_min,e_max,e_num,a_min,a_max,a_num = settings
    e_num = int(e_num); a_num = int(a_num); N = int(N)

    e_vals = np.linspace(e_min,e_max,e_num,endpoint=True)
    a_ratio = np.linspace(a_min,a_max,a_num,endpoint=True)[slice_index]

    to_plot = matrix[:,slice_index,:]

    for ax_index,ax in enumerate(np.reshape(axs,(-1))):
        ax.plot(e_vals,to_plot[ax_index,:],'.')
        ax.set_ylabel(label_key[ax_index])
        ax.set_xlabel('e_dd')

    fig.suptitle(f'Parameter minima as a function of e_dd (f = {f} Hz, {N} atoms, aspect ratio = {a_ratio:.3f})')
    plt.show()

def plot_2d(mat,settings,mode,nticksx=11,nticksy=11):
    f,N,e_min,e_max,e_num,a_min,a_max,a_num = settings
    e_num = int(e_num); a_num = int(a_num); N = int(N)

    fig, ax = plt.subplots()
    im = ax.imshow(mat[mode],vmax=None)

    nticksx = min(nticksx,e_num)
    nticksy = min(nticksy,a_num)
    ticksx = np.linspace(0,e_num-1,nticksx)
    ticksy = np.linspace(0,a_num-1,nticksy)
    tlabelsx = np.linspace(e_min,e_max,nticksx)
    tlabelsy = np.linspace(a_min,a_max,nticksy)

    ax.set_xticks(ticksx)
    ax.set_yticks(ticksy)
    ax.set_xticklabels(["{:.2f}".format(label) for label in tlabelsx])
    ax.set_yticklabels(["{:.2f}".format(label) for label in tlabelsy])

    ax.set_xlabel('e_dd')
    ax.set_ylabel('omega_z/omega_r')
    ax.set_aspect(e_num/a_num)

    fig.colorbar(im)
    fig.suptitle(f'{label_key[mode]}',fontsize='large')
    ax.set_title(f'gridsize: ({int(e_num)}x{int(a_num)}), f = {f} Hz, {N} atoms',fontsize='medium')
    
    plt.show()

def animate_evolution(mat,settings,slice_index,save=False,fps = 20):
    frame_len = 1/fps
    f,N,e_min,e_max,e_num,a_min,a_max,a_num = settings
    e_num = int(e_num); a_num = int(a_num); N = int(N)

    

    pass

label_key = {
    0: 'wavefunction contrast',
    1: 'energy / QHO energy',
    2: 'log10(decay time / s)',
    3: 'droplet number',
    4: 'transverse obliquity',
    5: 'transverse length / QHO length',
    6: 'droplet width / QHO length',
    7: 'droplet separation / QHO length',
    8: '1st order relative height',
    9: '2nd order relative height'
}

if __name__ == '__main__':
    mat = np.load('output\\outMat.npy')
    settings = np.load('output\\settings.npy')
    plot_1d(mat,settings,1)