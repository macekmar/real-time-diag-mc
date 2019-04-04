import numpy as np
from numpy import fft
from utility import vcut

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

def _fft(t, ft, n=None, axis=-1):
    r"""
    $f(\omega) = \int{dt} f(t) e^{i\omega t}$
    t is assumed sorted and regularly spaced
    """

    assert (len(t) == ft.shape[axis]), "coordinates should have the same length as values array on specified `axis`."
    if n is None:
        n = len(t)

    dt = t[1] - t[0]

    w = fft.fftshift(fft.fftfreq(n, dt))
    fw = fft.fftshift(fft.fft(ft, n=n, axis=axis), axes=axis)
    fw = np.swapaxes(fw, -1, axis)

    w = -2 * np.pi * w[::-1]
    fw = fw[..., ::-1]
    fw[..., :] *= dt * np.exp(1j * w * t[0])

    return w, np.swapaxes(fw, -1, axis)

def _get_min_padding(nb_t, dt, dw):
    """
    Minimum padding to get a frequency spacing smaller than dw from a time spacing dt.
    Returns an int larger than nb_t for security, as we don't want to crop the data but to pad it.
    """
    n = int(2. * np.pi / float(dt * dw) + 0.5)
    n = max(nb_t, n) # for security, we don't want to crop the data but to pad it !
    return n

def fourier_transform(t, ft, w_window, nb_w, axis=-1):
    """
    Fourier transform a sampled function given by the values `ft` at times `t`.
    Returns values at at least `nb_w` frequencies inside the window `w_window`.
    Manage zero padding to ensure FFT yields enough frequencies for it.
    """
    w_min, w_max = w_window
    dw = (w_max - w_min) / float(nb_w)
    n = _get_min_padding(len(t), t[1] - t[0], dw)
    n = _next_regular(n)

    w, fw = _fft(t, ft, n=n, axis=axis)
    w, fw = vcut(w, fw, left=w_min, right=w_max, axis=axis)
    return w, fw

if __name__ == '__main__':

    ############# tests ###########

    mu = 2.
    alpha = 3. + 1.j
    x0 = 1.j - 2.
    plot = False

    ### test 1
    t = np.linspace(-15., 15., 1000) + 1e-4 # shift slightly to avoid symmetry
    ft = mu * np.exp(-alpha * (t - x0)**2)
    t_ref = t.copy()
    ft_ref = ft.copy()

    def fw_ref(w):
        conv = -1
        return mu * np.exp(-1.j * conv * w * x0 - w**2 / (4 * alpha)) * np.sqrt(np.pi / alpha)

    w, fw = _fft(t, ft)
    assert np.allclose(fw, fw_ref(w), rtol=1e-10, atol=1e-12)

    ### test 2
    t = np.linspace(-15., 15., 1000) + 1e-4 # shift slightly to avoid symmetry
    ft = mu * np.exp(-alpha * (t - x0)**2)
    t_ref = t.copy()
    ft_ref = ft.copy()

    def fw_ref(w):
        conv = -1
        return mu * np.exp(-1.j * conv * w * x0 - w**2 / (4 * alpha)) * np.sqrt(np.pi / alpha)

    w, fw = fourier_transform(t, ft, [-2., 2.], 100)
    assert np.allclose(fw, fw_ref(w), rtol=1e-10, atol=1e-12)
    assert np.min(w) >= -2.
    assert np.max(w) <= 2.
    assert len(w) >= 100
    assert len(fw) == len(w)

    print 'Success !'
