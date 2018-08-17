import numpy as np
from toolbox import fourier_transform, vcut, symmetrize
import warnings
from copy import copy
from solver import expand_axis

def mult_by_1darray(a, b, axis):
    """
    Multiply element wize the ND array `a` by the 1D array `b` along `axis`.
    `axis` is an axis of `a`.
    Returns a ND array with same shape as `a`.
    """
    dim_array = np.ones((a.ndim,), dtype=int)
    dim_array[axis] = -1
    return a * b.reshape(dim_array)

def _fourier_transform_with_diracs(times, values, dirac_times, dirac_values, max_puls, min_data_pts):
    p = np.log2(np.pi*min_data_pts / float((times[1] - times[0]) * max_puls))
    n = 2**(int(p) + 1)

    while n < len(times):
        n *= 2

    if n > 1e7:
        warnings.warn('n={0} is huge, Fourier transform may cause memory overflow. Use less data points or a larger pulsation range.'.format(n), RuntimeWarning)

    freq, spectrum = fourier_transform(times, values, n=n, axis=1)
    puls = 2 * np.pi * freq
    puls, spectrum = vcut(puls, spectrum, -max_puls, max_puls, axis=1)

    for k in range(len(dirac_times)):
        spectrum += np.rollaxis(dirac_values[:, k, ..., np.newaxis] * np.exp(1j * puls[:] * dirac_times[k]), -1, 1)

    return puls, spectrum

def fiftyfifty(t, a, b):
    """
    If `t` is the coordinate array of the ND arrays `a` and `b` along their
    *second* dimension, this function returns the concatenation of `a`
    restricted to the negative part of `t` and of `b` restricted to its
    positive part.
    If t is zero at some index, then the mean of `a` and `b` values are taken.

    `t` is an increasingly sorted 1D array.
    `a` and `b` are ND arrays with N >= 2, they have the same shape and
    shape[1] = len(t).
    """
    i0 = np.searchsorted(t, 0.)
    output = np.concatenate((a[:, :i0], b[:, i0:]), axis=1)
    if t[i0] == 0.:
        output[:, i0] = 0.5 * (a[:, i0] + b[:, i0])
    return output

###############################################################################

class TimeGF(object):
    """
    A perturbation series as a function of time represented by a sampling along
    a time coordinate array. Dirac deltas can be added with a separate time
    coordinate array (not necessarily sorted). The function is assumed known on
    the negative, positive or both halves of the time axis.

    `values` is the samples array, it has shape (#orders, #times [, #orbitals]
    [, others...]), where #times is the size of the 1D array `times`.

    `dirac_values` has same shape as `values` except for the time axis which
    match the size of the 1D array `dirac_times`.

    `half` is one of the strings 'positive', 'negative', 'both'.
    """

    def __init__(self, times, values, dirac_times, dirac_values, half):
        self.times = times
        self.values = values # (order, times, orbitals, [others])
        self.dirac_times = dirac_times
        self.dirac_values = dirac_values # (order, times, orbitals, [others])
        self.half = half

        self._check_consistency()

    def _check_consistency(self):
        if len(self.times) != self.values.shape[1]:
            raise ValueError, 'Size of time coord does not match size of values.'

        if self.dirac_times is None or self.dirac_values is None or len(self.dirac_times) == 0:
            self.dirac_times = np.array([])
            dirac_shape = list(self.values.shape)
            dirac_shape[1] = 0
            self.dirac_values = np.array([], dtype=complex).reshape(dirac_shape)
        else:
            if len(self.dirac_times) != self.dirac_values.shape[1]:
                raise ValueError, 'Size of dirac time coord does not match size of dirac values.'
            shape_v = list(self.values.shape)
            shape_dv = list(self.dirac_values.shape)
            shape_v[1] = 0
            shape_dv[1] = 0
            if shape_v != shape_dv:
                raise ValueError, 'Shape of dirac values does not match shape of values.'

        if self.half not in ['negative', 'positive', 'both']:
            raise ValueError

    def fourier_transform(self, max_puls, min_data_pts=300):
        if self.half != 'both':
            warnings.warn(RuntimeWarning, 'The function is not completely known, Fourier transform is only partial!')

        puls, spectrum = _fourier_transform_with_diracs(self.times, self.values,
                self.dirac_times, self.dirac_values, max_puls=max_puls, min_data_pts=min_data_pts)

        return PulsGF(puls, spectrum)

    def increment_order(self, new_g0, new_dirac_times=None, new_dirac_values=None):
        """
        Promote the perturbation series to next order. The thus freed order
        zero is filled thanks to the function `new_g0` and the arrays
        `new_dirac_times` and `new_dirac_values`.

        Orbitals not implemented yet
        """
        new_values = new_g0(self.times)[np.newaxis, :]
        while new_values.ndim != self.values.ndim:
            new_values = new_values[..., np.newaxis]
        new_values = new_values * np.ones(self.values.shape[1:])
        self.values = np.concatenate((new_values, self.values), axis=0)

        if new_dirac_times is not None and new_dirac_values is not None:
            raise NotImplementedError
            for t, v in enumerate(new_dirac_times, new_dirac_values):
                if t not in self.dirac_times:
                    self.dirac_times = np.append(self.dirac_times, t)
        else:
            self.dirac_values = expand_axis(self.dirac_values, 0., axis=0)

        self._check_consistency()

    def apply_gf_sym(self):
        """
        Apply symmetry f_{ij}(t)^* = -f_{ji}(-t).

        Orbitals not implemented yet
        """
        if self.half == 'both':
            raise RuntimeError, 'The function is already fully known. Cannot apply symmetry.'

        new_times, sym_values = symmetrize(self.times, self.values, 0., axis=1)

        do_dirac = len(self.dirac_times) != 0
        if do_dirac:
            new_dirac_times, sym_dirac_values = symmetrize(self.dirac_times, self.dirac_values, 0., axis=1)

        if self.half == 'negative':
            new_values = fiftyfifty(new_times, sym_values, -np.conj(sym_values))
            if do_dirac:
                new_dirac_values = fiftyfifty(new_dirac_times, sym_dirac_values, -np.conj(sym_dirac_values))
        else:
            new_values = fiftyfifty(new_times, -np.conj(sym_values), sym_values)
            if do_dirac:
                new_dirac_values = fiftyfifty(new_dirac_times, -np.conj(sym_dirac_values), sym_dirac_values)

        if do_dirac:
            self.__init__(new_times, new_values, new_dirac_times, new_dirac_values, half='both')
        else:
            self.__init__(new_times, new_values, None, None, half='both')

###############################################################################

class PulsGF(object):
    """
    A perturbation series as a function of pulsation represented by a sampling
    along a pulsation coordinates array.

    `values` has shape (#orders, #puls [, #orbitals] [, others...]).
    """

    def __init__(self, puls, values):
        if len(puls) != values.shape[1]:
            raise ValueError, 'Size of puls coord does not match size of values.'
        self.puls = puls
        self.values = values # (order, puls, orbitals, [others])

    def mult_g_right(self, g0):
        """ Orbitals not implemented yet"""
        return PulsGF(self.puls,
                      mult_by_1darray(self.values, g0(self.puls), axis=1))

    def mult_g_left(self, g0):
        """ Orbitals not implemented yet"""
        return PulsGF(self.puls,
                      mult_by_1darray(self.values, g0(self.puls), axis=1))

    def increment_order(self, new_g0):
        """ Orbitals not implemented yet
        do not support part.
        """
        self.values = np.concatenate((new_g0(self.puls)[np.newaxis, :], self.values), axis=0)

###############################################################################

def is_incr_reg_spaced(a, rtol=1e-05, atol=1e-08):
    diff = a[1:] - a[:-1]
    is_ok = (np.abs(diff - diff[0]) < atol + rtol * np.abs(diff)).all()
    is_ok = is_ok and (diff > 0.).all()
    return is_ok

def theta(x):
    output = np.array(x > 0, dtype=float)
    output[x == 0] = 0.5
    return output

def convolve(a, b, mode, axis):
    """
    Convolves a ND array `a` with an 1D array `b` along one axis.
    """
    return np.apply_along_axis(lambda m: np.convolve(m, b, mode=mode), axis=axis, arr=a)

def convolve_coord(coord_a, coord_b, mode='full'):

    ### order inputs so that a is larger than b
    if len(coord_a) < len(coord_b):
        return convolve_coord(coord_b, coord_a, mode)

    ### check validity of coordinates arrays
    if not is_incr_reg_spaced(coord_a) or not is_incr_reg_spaced(coord_b):
        raise ValueError
    delta_t = coord_a[1] - coord_a[0]
    delta_t_b = coord_b[1] - coord_b[0]
    if np.abs(delta_t - delta_t_b) >= 1e-10 * delta_t:
        raise ValueError

    ### full coord array
    conv_coord = np.arange(coord_a[0] + coord_b[0], coord_a[-1] + coord_b[-1] + 0.5*delta_t, delta_t)

    if mode == 'full':
        return conv_coord
    elif mode == 'same':
        n = (len(coord_b)-1) // 2
        p = (len(coord_b)-1) % 2
        return conv_coord[n:-n-p]
    elif mode == 'valid':
        n = len(coord_b) - 1
        return conv_coord[n:-n]
    else:
        raise ValueError

class TimeKeldyshGF(object):

    def __init__(self, less, grea):
        """ should have same coordinates"""
        self.less = less # type: TimeGF
        self.grea = grea # type: TimeGF
        self.half = copy(less.half)
        self.times = self.less.times
        self.dirac_times = self.less.dirac_times

        self._check_consistency()

    def _check_consistency(self):
        if not self.less.half == self.grea.half:
            raise ValueError, 'Lesser and greater must be known on the same part.'
        if not np.array_equal(self.less.times, self.grea.times):
            raise ValueError, 'Time coordinates must match.'
        if not np.array_equal(self.less.dirac_times, self.grea.dirac_times):
            raise ValueError, 'Dirac time coordinates must match.'


    def fourier_transform(self, max_puls, min_data_pts=300):
        return PulsKeldyshGF(self.less.fourier_transform(max_puls, min_data_pts),
                             self.grea.fourier_transform(max_puls, min_data_pts))

    def retarded(self):
        """
        Compute the corresponding retarded function:
        reta(t) = theta(t)*(greater(t) - lesser(t))
        """
        if self.half == 'negative':
            raise RuntimeError, 'Lesser and greater must be known on the positive half to do this operation.'

        reta_values = self.grea.values - self.less.values
        reta_values[self.times < 0.] = 0.
        reta_values[self.times == 0.] *= 0.5

        reta_dirac_values = self.grea.dirac_values - self.less.dirac_values
        reta_dirac_values[self.dirac_times < 0.] = 0.
        reta_dirac_values[self.dirac_times == 0.] *= 0.5

        return TimeGF(self.times, reta_values, self.dirac_times, reta_dirac_values, 'both')

    def convol_g_right(self, g0_less, g0_grea, cutoff_g0):
        """
        Compute the Keldysh right-hand-side convolution with g0: G = K*g0.
        The function needs to be known on the positive times, and the result is
        the convoluted function on the positive times only.

        g0 must is given through a lesser and greater *vectorized* functions of
        time, with *no window*.
        `cutoff_g0` is used to limit the array of g0 values used in the
        convolution. However, values outside of this window can be requested
        for dirac deltas. The larger is `cutoff_g0`, the larger is the result
        time coord array.

        Orbitals not implemented yet
        """
        if self.half == 'negative':
            raise RuntimeError, 'Lesser and greater must be known on the positive half to do this operation.'
        if not is_incr_reg_spaced(self.times):
            raise RuntimeError, 'Time coordinates should be an increasing and regularly spaced float array.'

        times = self.times
        delta_t = times[1] - times[0]
        times_g0 = np.arange(-(cutoff_g0 // delta_t), cutoff_g0 // delta_t + 0.5, 1.) * delta_t
        if len(times_g0) < len(times):
            raise ValueError, 'Need a larger cutoff.'
        new_times = convolve_coord(times, times_g0, mode='same')
        mask = new_times >= 0.
        new_times = new_times[mask]

        g0_less_values = g0_less(times_g0)
        g0_grea_values = g0_grea(times_g0)
        g0_ordr_values = theta(-times_g0) * g0_less_values + theta(times_g0) * g0_grea_values
        g0_anti_values = theta(-times_g0) * g0_grea_values + theta(times_g0) * g0_less_values

        less_values = convolve(self.less.values, g0_anti_values, mode='same', axis=1)
        less_values += convolve(self.grea.values, g0_less_values, mode='same', axis=1)
        less_values *= delta_t
        grea_values = convolve(self.grea.values, g0_ordr_values, mode='same', axis=1)
        grea_values += convolve(self.less.values, g0_grea_values, mode='same', axis=1)
        grea_values *= delta_t
        less_values = less_values[:, mask]
        grea_values = grea_values[:, mask]

        g0_ordr = lambda t: theta(-t) * g0_less(t) + theta(t) * g0_grea(t)
        g0_anti = lambda t: theta(-t) * g0_grea(t) + theta(t) * g0_less(t)
        for k, t in enumerate(self.dirac_times):
            if self.half == 'both' or t >= 0.: # FIXME: not sure what to do if t==0
                shifted_new_times = new_times - t
                less_values += self.grea.dirac_values[:, k:k+1] * g0_less(shifted_new_times)[np.newaxis, :]
                less_values += self.less.dirac_values[:, k:k+1] * g0_anti(shifted_new_times)[np.newaxis, :]
                grea_values += self.grea.dirac_values[:, k:k+1] * g0_ordr(shifted_new_times)[np.newaxis, :]
                grea_values += self.less.dirac_values[:, k:k+1] * g0_grea(shifted_new_times)[np.newaxis, :]

        lesser = TimeGF(new_times, less_values, None, None, half='positive')
        greater = TimeGF(new_times, grea_values, None, None, half='positive')
        return TimeKeldyshGF(lesser, greater)

    def convol_g_left(self, g0_less, g0_grea, cutoff_g0):
        """
        Compute the Keldysh left-hand-side convolution with g0: G = g0*K.
        The function needs to be known on the negative times, and the result is
        the convoluted function on the negative times only.

        See doc of convol_g_right for more details.

        Orbitals not implemented yet
        """
        if self.half == 'positive':
            raise RuntimeError, 'Lesser and greater must be known on the negative half to do this operation.'
        if not is_incr_reg_spaced(self.times):
            raise RuntimeError, 'Time coordinates should be an increasing and regularly spaced float array.'

        times = self.times
        delta_t = times[1] - times[0]
        times_g0 = np.arange(-(cutoff_g0 // delta_t), cutoff_g0 // delta_t + 0.5, 1.) * delta_t
        if len(times_g0) < len(times):
            raise ValueError, 'Need a larger cutoff.'
        new_times = convolve_coord(times, times_g0, mode='same')
        mask = new_times <= 0.
        new_times = new_times[mask]

        g0_less_values = g0_less(times_g0)
        g0_grea_values = g0_grea(times_g0)
        g0_ordr_values = theta(-times_g0) * g0_less_values + theta(times_g0) * g0_grea_values
        g0_anti_values = theta(-times_g0) * g0_grea_values + theta(times_g0) * g0_less_values

        less_values = convolve(self.less.values, g0_ordr_values, mode='same', axis=1)
        less_values += convolve(self.grea.values, g0_less_values, mode='same', axis=1)
        less_values *= delta_t
        grea_values = convolve(self.less.values, g0_grea_values, mode='same', axis=1)
        grea_values += convolve(self.grea.values, g0_anti_values, mode='same', axis=1)
        grea_values *= delta_t
        less_values = less_values[:, mask]
        grea_values = grea_values[:, mask]

        g0_ordr = lambda t: theta(-t) * g0_less(t) + theta(t) * g0_grea(t)
        g0_anti = lambda t: theta(-t) * g0_grea(t) + theta(t) * g0_less(t)
        for k, t in enumerate(self.dirac_times):
            if self.half == 'both' or t <= 0.: # FIXME: not sure what to do if t==0
                shifted_new_times = new_times - t
                less_values += g0_ordr(shifted_new_times)[np.newaxis, :] * self.less.dirac_values[:, k:k+1]
                less_values += g0_less(shifted_new_times)[np.newaxis, :] * self.grea.dirac_values[:, k:k+1]
                grea_values += g0_grea(shifted_new_times)[np.newaxis, :] * self.less.dirac_values[:, k:k+1]
                grea_values += g0_anti(shifted_new_times)[np.newaxis, :] * self.grea.dirac_values[:, k:k+1]

        lesser = TimeGF(new_times, less_values, None, None, half='negative')
        greater = TimeGF(new_times, grea_values, None, None, half='negative')
        return TimeKeldyshGF(lesser, greater)

    def increment_order(self, new_g0_less, new_g0_grea, new_less_d_times=None, new_less_d_values=None,
                              new_grea_d_times=None, new_grea_d_values=None):
        """ Orbitals not implemented yet"""
        self.less.increment_order(new_g0_less, new_less_d_times, new_less_d_values)
        self.grea.increment_order(new_g0_grea, new_grea_d_times, new_grea_d_values)

    def apply_part_hole_sym(self):
        """
        Apply symmetry lesser_{ij}(t) = -greater_{ji}(-t).

        Orbitals not implemented yet
        """
        ### FIXME: do dirac or not
        if self.half == 'both':
            raise RuntimeError, 'The function is already fully known. Cannot apply symmetry.'

        new_times, sym_less = symmetrize(self.times, self.less.values, 0., axis=1)
        new_times, sym_grea = symmetrize(self.times, self.grea.values, 0., axis=1)

        new_dirac_times, sym_dirac_less = symmetrize(self.dirac_times, self.less.dirac_values, 0., axis=1)
        new_dirac_times, sym_dirac_grea = symmetrize(self.dirac_times, self.grea.dirac_values, 0., axis=1)

        if self.half == 'negative':
            new_less_values = fiftyfifty(new_times, sym_less, -sym_grea)
            new_grea_values = fiftyfifty(new_times, sym_grea, -sym_less)
            new_less_dirac_values = fiftyfifty(new_dirac_times, sym_dirac_less, -sym_dirac_grea)
            new_grea_dirac_values = fiftyfifty(new_dirac_times, sym_dirac_grea, -sym_dirac_less)
        else:
            new_less_values = fiftyfifty(new_times, -sym_grea, sym_less)
            new_grea_values = fiftyfifty(new_times, -sym_less, sym_grea)
            new_less_dirac_values = fiftyfifty(new_dirac_times, -sym_dirac_grea, sym_dirac_less)
            new_grea_dirac_values = fiftyfifty(new_dirac_times, -sym_dirac_less, sym_dirac_grea)

        new_less = TimeGF(new_times, new_less_values, new_dirac_times, new_less_dirac_values, half='both')
        new_grea = TimeGF(new_times, new_grea_values, new_dirac_times, new_grea_dirac_values, half='both')

        self.__init__(new_less, new_grea)

    def apply_gf_sym(self):
        """
        Apply symmetry lesser_{ij}(t)^* = -lesser_{ji}(-t), idem for greater.

        Orbitals not implemented yet
        """
        if self.half == 'both':
            raise RuntimeError, 'The function is already fully known. Cannot apply symmetry.'

        self.less.apply_gf_sym()
        self.grea.apply_gf_sym()

        self.times = self.less.times
        self.dirac_times = self.less.dirac_times
        self.half = self.less.half
        self._check_consistency()

###############################################################################

class PulsKeldyshGF(object):

    def __init__(self, less, grea):
        self.less = less # type: PulsGF
        self.grea = grea # type: PulsGF
        self.puls = self.less.puls

        self._check_consistency()

    def _check_consistency(self):
        if not np.array_equal(self.less.puls, self.grea.puls):
            raise ValueError, 'Lesser and greater should have the same coordinates array.'

    def increment_order(self, new_g0_less, new_g0_grea):
        """ Orbitals not implemented yet"""
        self.less.increment_order(new_g0_less)
        self.grea.increment_order(new_g0_grea)

###############################################################################

def kernel_GF_from_archive(ar, name, nonfixed_op):
    kernels = mult_by_1darray(ar[name]['kernels'], ar[name]['cn'][1:], axis=0)
    kernel_diracs = mult_by_1darray(ar[name]['kernel_diracs'], ar[name]['cn'][1:], axis=0)

    a = 1 if nonfixed_op else 0

    less = TimeGF(ar[name]['bin_times'],
                  kernels[:, :, a],
                  ar[name]['dirac_times'],
                  kernel_diracs[:, :, a],
                  'negative')

    a = 1 - a
    grea = TimeGF(ar[name]['bin_times'],
                  kernels[:, :, a],
                  ar[name]['dirac_times'],
                  kernel_diracs[:, :, a],
                  'negative')

    return TimeKeldyshGF(less, grea)

if __name__ == '__main__':

    ### test TimeGF.increment_order
    times = np.linspace(0, 1, 10)
    values = np.zeros((2, 10, 3)) # 2 orders, 10 times, 3 parts
    dirac_times = np.array([0.1])
    dirac_values = np.ones((2, 1, 3))
    f = TimeGF(times, values, dirac_times, dirac_values, half='positive')
    g0 = np.vectorize(lambda t: 0.5)
    f.increment_order(g0, None, None)
    assert f.values.shape == (3, 10, 3)
    assert f.dirac_values.shape == (3, 1, 3)
    assert (f.values[0, :, :] == 0.5).all() # order 0 is g0
    assert (f.values[1:, :, :] == 0.).all() # other orders untouched
    assert (f.dirac_values[0, :, :] == 0.).all() # no dirac in order 0
    assert (f.dirac_values[1:, :, :] == 1.).all() # other orders untouched

    ### test TimeGF.increment_order
    if False:
        times = np.linspace(0, 1, 10)
        values = np.zeros((2, 10, 3)) # 2 orders, 10 times, 3 parts
        dirac_times = np.array([0.1])
        dirac_values = np.ones((2, 1, 3))
        f = TimeGF(times, values, dirac_times, dirac_values, half='positive')
        g0 = np.vectorize(lambda t: 0.5)
        f.increment_order(g0, np.array([0.1, 0.5]), 0.2*np.ones((2, 3)))
        assert f.values.shape == (3, 10, 3)
        assert f.dirac_values.shape == (3, 2, 3)
        assert (f.values[0, :, :] == 0.5).all() # order 0 is g0
        assert (f.values[1:, :, :] == 0.).all() # other orders untouched
        assert (f.dirac_times == np.array([0.1, 0.5])).all()
        assert (f.dirac_values[0, :, :] == 0.2).all() # no dirac in order 0
        assert (f.dirac_values[1:, 0, :] == 1.).all() # other orders untouched
        assert (f.dirac_values[1:, 1, :] == 0.).all() # other orders untouched

    ### test convolve_coord
    ta = np.arange(0., 5. + 0.1, 0.5) # +0.1 to include 5.
    tb = np.arange(-2.2, 1.8 + 0.1, 0.5)
    a = np.linspace(0, 1, len(ta))
    b = np.linspace(0, 1, len(tb))
    tab = convolve_coord(ta, tb, mode='full')
    ab = np.convolve(a, b, mode='full')
    assert len(tab) == len(ab)
    assert np.array_equal(tab, np.arange(-2.2, 6.8 + 0.1, 0.5))
    tab = convolve_coord(ta, tb, mode='same')
    ab = np.convolve(a, b, mode='same')
    assert len(tab) == len(ab)
    tab = convolve_coord(ta, tb, mode='valid')
    ab = np.convolve(a, b, mode='valid')
    assert len(tab) == len(ab)

    ### test convolve_coord
    ### def ta and a
    ta = np.linspace(-1., 1., 11)
    a = np.zeros(11)
    a[np.where(ta == 0.)[0]] = 1. # a is a peak centered on ta = 0
    ### def tb and b
    tb = np.linspace(-1., 1.2, 12) # = ta plus an extra point on the right
    assert (tb[:-1] == ta).all()
    b = np.zeros(12)
    b[np.where(tb == 0.)[0]] = 1. # b is a peak centered on tb = 0
    ### def tc and c
    tc = np.linspace(-1.2, 1.2, 13) # = ta plus an extra point on the right and on the left
    assert np.isclose(tc[1:-1], ta).all()
    c = np.zeros(13)
    c[np.where(tc == 0.)[0]] = 1. # c is a peak centered on tc = 0
    for mode in ['full', 'same', 'valid']:
        ### for different input array sizes, check if autoconvol of a centered
        ### peak gives a centered peak (it should)
        # print mode
        tol = 1e-10

        aa = np.convolve(a, a, mode=mode)
        taa = convolve_coord(ta, ta, mode=mode)
        assert len(taa) == len(np.convolve(ta, ta, mode=mode))
        assert np.abs(taa[np.where(aa == 1.)[0]]) < tol

        if mode != 'valid': # peak is not visible in valid mode here
            bb = np.convolve(b, b, mode=mode)
            tbb = convolve_coord(tb, tb, mode=mode)
            assert len(tbb) == len(np.convolve(tb, tb, mode=mode))
            assert np.abs(tbb[np.where(bb == 1.)[0]]) < tol

        ab = np.convolve(a, b, mode=mode)
        tab = convolve_coord(ta, tb, mode=mode)
        assert len(tab) == len(np.convolve(ta, tb, mode=mode))
        assert np.abs(tab[np.where(ab == 1.)[0]]) < tol

        cb = np.convolve(c, b, mode=mode)
        tcb = convolve_coord(tc, tb, mode=mode)
        assert len(tcb) == len(np.convolve(tc, tb, mode=mode))
        assert np.abs(tcb[np.where(cb == 1.)[0]]) < tol
