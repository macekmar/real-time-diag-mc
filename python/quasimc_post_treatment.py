import numpy as np
from post_treatment import _centered
from fourier_transform import fourier_transform, _fft, _next_regular
from scipy.fftpack import ifft
from utility import vcut

def add_empty_orders(arr, orders):
    """Extend arr with zeros to hold all orders.

    Array `arr` only contains the given `orders`. This extends it with zeros, 
    so that it contains all orders from 1 up to `max(orders)`."""
    ord_max = max(orders)
    shp = list(arr.shape)
    shp[0] = ord_max
    new_arr = np.zeros(shp, dtype=arr.dtype)
    for io, order in enumerate(orders):
        new_arr[order-1,...] = arr[io,...]
    return new_arr

def g0_w_func(w, epsilon_d, Gamma):
    return 1/(w - epsilon_d + 1j*Gamma)

def get_ret_kernels_w(times, kernels, w_window, nb_w):
    KR = kernels[:,:,0,:] + kernels[:,:,1,:]
    KR = np.conj(KR)[:,::-1,:]
    times = -times[::-1]

    return fourier_transform(times, KR, w_window, nb_w, axis=1)


def get_GR_w(times, kernels, orders, w_window, nb_w, epsilon_d, Gamma):
    w, KR_w = get_ret_kernels_w(times, kernels, w_window, nb_w)
    g0_w = g0_w_func(w, epsilon_d, Gamma)

    GR_w = np.einsum(r't...,it...->it...', g0_w, KR_w)
    # `*` is along axis, we want to multiply along the first axis
    GR_w = (Gamma**(orders+1) * GR_w.T).T
    return w, GR_w


if __name__ == '__main__':
    print 'Start tests'

    # ### Test add_empty_orders

    orders = [2,3,6]

    shp = (len(orders), 10000, 2, 1)
    kernels = np.random.rand(*shp) + 1j*np.random.rand(*shp)
    new_shp = (max(orders), 10000, 2, 1)
    expanded_kernels = add_empty_orders(kernels, orders)

    assert expanded_kernels.shape == new_shp
    assert np.allclose(np.sum(kernels), np.sum(expanded_kernels))
    for io, order in enumerate(orders):
        assert np.allclose(expanded_kernels[order-1], kernels[io])
    

    # Different shape
    shp = (3, 10000, 2, 1, 10)
    kernels_part = np.random.rand(*shp) + 1j*np.random.rand(*shp)
    new_shp = (max(orders), 10000, 2, 1, 10)
    expanded_kernels_part = add_empty_orders(kernels_part, orders)
    
    assert expanded_kernels_part.shape == new_shp
    assert np.allclose(np.sum(kernels_part), np.sum(expanded_kernels_part))
    for io, order in enumerate(orders):
        assert np.allclose(expanded_kernels_part[order-1], kernels_part[io])