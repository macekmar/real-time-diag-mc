import numpy as np
from scipy.fftpack import ifft

from fourier_transform import _fft, _next_regular, fourier_transform
from post_treatment import _centered
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
