import numpy as np
def f_x4m(N,L,in_vect):
    """Approximates input as periodic sampled input.
    Input vector has N entries between +-L/2.
    Returns Fourier transformed array centred on zero to best match FT definition,
    and corresponding k's."""
    ks = 2*np.pi*np.fft.fftfreq(N,L/N)
    x4m = L/N * np.exp(ks*L/2*1j) * np.fft.fft(in_vect)
    x4m = np.fft.fftshift(x4m)
    return x4m, np.fft.fftshift(ks)

def inv_f_x4m(N,k_range,in_vect):
    """Inverse fourier transform. Same format as f_x4m; returns corresponding x's."""
    inv_input = in_vect[::-1]
    x4m = f_x4m(N,k_range,np.concatenate(([0],inv_input[:-1])))
    return 1/(2*np.pi)*x4m[0], x4m[1]

### TESTING ###
"""import matplotlib.pyplot as plt
L = 100
N = 20
zs = np.linspace(-L/2,L/2,N,endpoint=False,dtype=complex)

func = np.exp(-np.abs(zs-10))
transform, k_vals = f_x4m(N,L,func)
# works up to here. now try inverse transform.
og_func, x_vals = inv_f_x4m(N,k_vals[-1]-2*k_vals[0]+k_vals[1],transform)

plt.plot(zs,func)
plt.plot(x_vals, og_func)
plt.show()"""