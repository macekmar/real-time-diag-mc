import numpy as np
from scipy.fftpack import fft, ifft

#########   Generalized fft convolution #########

# stolen from scipy.signal.fftconvolve
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

# stolen from scipy.signal.fftconvolve
def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

# inspired from scipy.signal.fftconvolve
def generalized_fftconvolve(in1, in2, subscripts=r't, t -> t'):
    """
    Convolves `in1` with `in2` along their first axis using a product defined by `subscripts`, and taking advantage of the FFT algorithm.

    `subscripts` is an input of numpy.einsum used to define the product between the Fourier transforms of `in1` and `in2` along their first axis. Inputs shapes must match accordingly.

    `same` convolution mode only.
    """
    length = in1.shape[0] + in2.shape[0] - 1

    # Speed up FFT by padding to optimal size for FFTPACK
    opt_length = _next_regular(length)
    in1_fft = fft(in1, opt_length, axis=0)
    in2_fft = fft(in2, opt_length, axis=0)
    # print np.einsum_path(subscripts, in1_fft, in2_fft, optimize='optimal')[1]
    ret = ifft(np.einsum(subscripts, in1_fft, in2_fft), axis=0)

    return _centered(ret[:length].copy(), len(in1))

######### Post treatment functions #########


def compute_correlator(archive, g0_func, no_cn=False):
    """
    From an archive of QMC results containing kernel data, computes the (perturbation series of the) correlator associated by convolution.

    `archive` is a result archive created by the QMC with the kernel method.
    `g0_func` is the unperturbed one-particle Green's function. It is a function of time returning a scalar value (retarded/advanced), a 2D square array (keldysh matrix), or an array of shape (N, N, M, M) (keldysh matrix with orbitals).

    Note `g0_func` is NOT in general the unperturbed correlator, convolution has to be done with the *one-particle* unperturbed Green's function.

    TODO: add diracs !!
    TODO: optimize for big system sizes
    """

    if archive['parameters']['method'] == 0:
        raise ValueError, 'The archive contains non-kernel method results, cannot proceed.'

    if archive['parameters']['nonfixed_op'] == True:
        raise NotImplementedError # devpt along column, TODO

    def extract(ar):
        ### K.shape = (orders, times, Keldysh, [orbitals,] [part,])

        if no_cn:
            K = ar['kernels']
        else:
            if ar['cn'].ndim <= 1:
                K = np.einsum(r'i...,i->i...', ar['kernels'], ar['cn'][1:])
            elif ar['cn'].ndim == 2:
                K = np.einsum(r'i...j,ij->i...j', ar['kernels'], ar['cn'][1:])
            else:
                raise ValueError

        times = ar['bin_times']
        delta_t = times[1] - times[0]
        times_g0 = np.arange(times[0] - times[-1], times[-1] - times[0] + 0.5 * delta_t, delta_t)
        g0 = np.array([g0_func(t) for t in times_g0])

        if g0.ndim >= 3:
            if g0.shape[1] != g0.shape[2] or g0.shape[1] != K.shape[2]:
                raise ValueError
        if g0.ndim >= 5:
            if g0.shape[3] != g0.shape[4] or g0.shape[3] != K.shape[3]:
                raise ValueError

        G = np.empty_like(K)

        # if g0.ndim == 1:
        #     G = generalized_fftconvolve(K, g0, r'nt...,t->nt...') * delta_t
        # elif g0.ndim == 3:
        #     G = generalized_fftconvolve(K, g0, r'ntj...,tij->nti...') * delta_t
        # elif g0.ndim == 5:
        #     G = generalized_fftconvolve(K, g0, r'ntjl...,tijkl->ntik...') * delta_t
        # else:
        #     raise ValueError

        for n in range(len(K)):
            if g0.ndim == 1:
                G[n] = generalized_fftconvolve(K[n], g0, r't...,t->t...') * delta_t
            elif g0.ndim == 3:
                G[n] = generalized_fftconvolve(K[n], g0, r'tj...,tij->ti...') * delta_t
            elif g0.ndim == 5:
                G[n] = generalized_fftconvolve(K[n], g0, r'tjl...,tijkl->tik...') * delta_t
            else:
                raise ValueError

        return times, G

    times, G = extract(archive['results_all'])
    times_part, G_part = extract(archive['results_part'])

    return times, G, times_part, G_part


def compute_correlator_oldway(archive, abs_g0_value):
    """
    From an archive of QMC results containing non-kernel data, returns the (perturbation series of the) physical value associated.

    `archive` is a result archive created by the QMC with the oldway method.
    `abs_g0_value` is the absolute value of the unperturbed physical value.
    """

    if archive['parameters']['method'] > 0:
        raise ValueError, 'The archive contains kernel method results, cannot proceed.'

    abs_g0_value = np.abs(np.asscalar(abs_g0_value))

    def extract(ar):
        return ar['sn'] * ar['cn'] * abs_g0_value

    G = extract(archive['results_all'])
    G_part = extract(archive['results_part'])

    return G, G_part


######### Utilities #########

def make_g0_contour(g0_less, g0_grea):
    """
    Inputs are functions of time returning a scalar (no orbitals) or a 2D matrix value (with orbitals).
    """
    def g0_contour(t):
        less = np.squeeze(g0_less(t))
        grea = np.squeeze(g0_grea(t))
        if t < 0.:
            return np.array([[less, less], [grea, grea]], dtype=complex)
        elif t > 0.:
            return np.array([[grea, less], [grea, less]], dtype=complex)
        else: # t == 0.
            return np.array([[0.5 * (less + grea), less], [grea, 0.5 * (less + grea)]], dtype=complex)
    return g0_contour


if __name__ == '__main__':
    from scipy import signal

    ### test generalized_fftconvolve
    a = generalized_fftconvolve(np.arange(10), np.arange(5))
    a_ref = signal.fftconvolve(np.arange(10), np.arange(5), mode='same')
    assert a.shape == a_ref.shape
    assert np.allclose(a, a_ref)

    ### test generalized_fftconvolve
    a = np.arange(40).reshape((10, 2, 2))
    b = np.arange(10).reshape((5, 2))

    c_ref = np.array([[  225.,   315.],
                    [  405.,   495.],
                    [  585.,   675.],
                    [  765.,   855.],
                    [  945.,  1035.]])

    c = generalized_fftconvolve(b, a, 'tj,tij->ti')
    assert c.shape == c_ref.shape
    assert np.allclose(c, c_ref)
