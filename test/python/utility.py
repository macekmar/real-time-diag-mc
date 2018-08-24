import numpy as np

def cpx_interp(x, xp, fp, left=None, right=None, period=None):
    """Simple wrapping of numpy.interp for complex values"""
    r = np.interp(x, xp, fp.real, left, right, period)
    i = np.interp(x, xp, fp.imag, left, right, period)
    return r + 1j*i

from scipy import interpolate
def interp_1D(x, xp, fp, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate'):
    """Wrapping of scipy.interpolate.interp1d, so as to use it in the same fashion as np.interp."""
    func = interpolate.interp1d(xp, fp, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    return func(x)

def cpx_interp_1D(x, xp, fp, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate'):
    if fill_value != 'extrapolate':
        raise NotImplementedError
    r = interp_1D(x, xp, fp.real, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    i = interp_1D(x, xp, fp.imag, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    return r + 1j*i

