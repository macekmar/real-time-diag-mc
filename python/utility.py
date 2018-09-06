import numpy as np
from scipy import signal

def expand_axis(a, val, end=False, axis=0):
    """
    Expand the axis `axis` of array `a` with `val`.
    Returns an array with same shape as `a` but increased by one along axis `axis`.
    `val` is used to fill in the thus openned positions.
    """
    # FIXME: is this the same as np.insert(a, 0, val, axis=0) ???
    shape = list(a.shape)
    shape[axis] = 1
    if end:
        return np.append(a, np.tile(val, shape), axis=axis)
    else:
        return np.append(np.tile(val, shape), a, axis=axis)

def squeeze_except(a, except_axes):
    """
    Like np.squeeze but ignoring axes in the list `except_axis`.
    """
    shape = list(a.shape)
    for d in range(a.ndim):
        if d not in except_axes and shape[d] == 1:
            shape[d] = -1
    k = 0
    while k < len(shape):
        if shape[k] == -1:
            del shape[k]
        else:
            k += 1

    return a.reshape(tuple(shape))


def reduce_binning(x, chunk_size, axis=-1):
    """
    Reduces the size of ND array `x` along dimension `axis` by summing together
    chunks of size `chunk_size`.
    """
    x = np.swapaxes(x, axis, 0)
    shape = (-1, chunk_size) + x.shape[1:]
    x = x[:x.shape[0] - x.shape[0] % chunk_size].reshape(shape).sum(axis=1)
    return np.swapaxes(x, axis, 0)

def mult_by_1darray(a, b, axis):
    """
    Multiply element wize the ND array `a` by the 1D array `b` along `axis`.
    `axis` is an axis of `a`.
    Returns a ND array with same shape as `a`.
    """
    dim_array = np.ones((a.ndim,), dtype=int)
    dim_array[axis] = -1
    return a * b.reshape(dim_array)

def mult_by_2darray(a, b, axis1, axis2):
    """
    Multiply element wize the ND array `a` by the 2D array `b` along `axis1` and `axis2`.
    `axis1` and `axis2` are axes of `a`.
    Returns a ND array with same shape as `a`.
    """
    dim_array = np.ones((a.ndim,), dtype=int)
    dim_array[axis1] = b.shape[0]
    dim_array[axis2] = b.shape[1]
    return a * b.reshape(dim_array)

def is_incr_reg_spaced(a, rtol=1e-05, atol=1e-08):
    diff = a[1:] - a[:-1]
    is_ok = (np.abs(diff - diff[0]) < atol + rtol * np.abs(diff)).all()
    is_ok = is_ok and (diff > 0.).all()
    return is_ok

def convolve(a, b, mode, axis):
    """
    Convolves a ND array `a` with an 1D array `b` along one axis.
    """
    if a.shape[axis] < len(b):
        return np.apply_along_axis(lambda m: signal.fftconvolve(b, m, mode=mode), axis=axis, arr=a)
    else:
        return np.apply_along_axis(lambda m: signal.fftconvolve(m, b, mode=mode), axis=axis, arr=a)

def convolve_coord(coord_a, coord_b, mode='full'):
    """
    """

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

def vcut(coord, values, left=None, right=None, axis=-1):
    """
    Cut the coordinate and values arrays of a sampled function so as to reduce its coordinate range to [`left`, `right`].

    None means infinity.
    """
    coord_out = coord.copy()
    values_out = np.swapaxes(values.copy(), 0, axis)
    if left is not None:
        left_i = np.argmin(np.abs(coord_out - left))
        coord_out = coord_out[left_i:]
        values_out = values_out[left_i:]
    if right is not None:
        right_i = np.argmin(np.abs(coord_out - right)) + 1
        coord_out = coord_out[:right_i]
        values_out = values_out[:right_i]
    return coord_out, np.swapaxes(values_out, 0, axis)


if __name__ == '__main__':
    print 'Start tests'

    ### test reduce_binning
    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 5, 1), np.array([[10, 35, 60], [85, 110, 135]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 6, 1), np.array([[15, 51], [105, 141]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [0]).shape == (1, 5, 4)

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [1]).shape == (5, 4)

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [1, 2]).shape == (5, 1, 4)

    ### test squeeze_except
    a = np.array([]).reshape(0, 5, 1, 4)
    assert a.shape == (0, 5, 1, 4)
    assert squeeze_except(a, [1]).shape == (0, 5, 4)

    ### test mult_by_2darray
    a = np.ones((2, 3, 4))
    b = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]) # of shape (2, 4)
    res = np.array([[[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
                    [[5., 6., 7., 8.], [5., 6., 7., 8.], [5., 6., 7., 8.]]])
    assert np.array_equal(mult_by_2darray(a, b, 0, 2), res)

    def convolve_to_test(*args, **kwargs):
        kwargs['axis'] = 0
        return convolve(*args, **kwargs)
        # return np.convolve(*args, **kwargs)

    ### test convolve_coord
    ta = np.arange(0., 5. + 0.1, 0.5) # +0.1 to include 5.
    tb = np.arange(-2.2, 1.8 + 0.1, 0.5)
    a = np.linspace(0, 1, len(ta))
    b = np.linspace(0, 1, len(tb))
    tab = convolve_coord(ta, tb, mode='full')
    ab = convolve_to_test(a, b, mode='full')
    assert len(tab) == len(ab)
    assert np.array_equal(tab, np.arange(-2.2, 6.8 + 0.1, 0.5))
    tab = convolve_coord(ta, tb, mode='same')
    ab = convolve_to_test(a, b, mode='same')
    assert len(tab) == len(ab)
    tab = convolve_coord(ta, tb, mode='valid')
    ab = convolve_to_test(a, b, mode='valid')
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

        aa = convolve_to_test(a, a, mode=mode)
        taa = convolve_coord(ta, ta, mode=mode)
        assert len(taa) == len(convolve_to_test(ta, ta, mode=mode))
        assert np.abs(taa[np.where(aa == 1.)[0]]) < tol

        if mode != 'valid': # peak is not visible in valid mode here
            bb = convolve_to_test(b, b, mode=mode)
            tbb = convolve_coord(tb, tb, mode=mode)
            assert len(tbb) == len(convolve_to_test(tb, tb, mode=mode))
            assert np.abs(tbb[np.where(bb == 1.)[0]]) < tol

        ab = convolve_to_test(a, b, mode=mode)
        tab = convolve_coord(ta, tb, mode=mode)
        assert len(tab) == len(convolve_to_test(ta, tb, mode=mode))
        assert np.abs(tab[np.where(ab == 1.)[0]]) < tol

        cb = convolve_to_test(c, b, mode=mode)
        tcb = convolve_coord(tc, tb, mode=mode)
        assert len(tcb) == len(convolve_to_test(tc, tb, mode=mode))
        assert np.abs(tcb[np.where(cb == 1.)[0]]) < tol

    print 'Success'
