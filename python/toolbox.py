import numpy as np
import matplotlib.pyplot as plt
from math import pi
import copy

######## Archive HDF5 ########
import datetime

def hdf5_to_dict(archive, print_groups=False, prefix='> '):
    primitive_types = (int, float, complex, list, tuple, str)
    data = {}

    try:
        iter(archive) # is it iterable ?
    except TypeError:
        print prefix + ' (' + type(archive).__name__ + ')'
        return archive

    for key in archive:
        if isinstance(archive[key], primitive_types) or archive.is_data(key):
            if print_groups:
                print prefix + key + ' (' + type(archive[key]).__name__ + ')'
            data[key] = archive[key]
        elif archive.is_group(key):
            if print_groups:
                print prefix + key
            data[key] = hdf5_to_dict(archive[key], print_groups, prefix + '> ')
        else:
            raise RuntimeError
    return data

def seconds_to_time(seconds):
    return str(datetime.timedelta(seconds=seconds))

######## Archive HDF5 : CtintKeldysh ########

def check_staircase(staircase_list):
    if len(staircase_list) == 0:
        return
    assert(len(staircase_list[0]) > 0) # no empty array
    for i in range(len(staircase_list)-1):
        assert(len(staircase_list[i+1]) > len(staircase_list[i])) # TODO: >=

def staircase_leading_elts(staircase_list):
    check_staircase(staircase_list)
    output = []
    k = 0
    for array in staircase_list:
        while k < len(array):
            output.append(array[k])
            k += 1

    return np.array(output)

######## Matrix ########

def antitranspose(m):
    assert(len(m.shape) >= 2)
    assert(m.shape[0] == m.shape[1])
    assert(m.shape[0] == 2)
    out = m.copy()
    out[0, 0] = m[1, 1].copy()
    out[1, 1] = m[0, 0].copy()
    return out

######## Plotting ########

def cpx_plot(ax, coord, values, *args, **kwargs):
    args = list(args)
    re_args = copy.copy(args)
    im_args = copy.copy(args)
    for i in range(len(args)):
        if isinstance(args[i], tuple):
            re_args[i], im_args[i] = args[i]
    re_args = tuple(re_args)
    im_args = tuple(im_args)

    re_kwargs = kwargs.copy()
    im_kwargs = kwargs.copy()
    for kw in kwargs:
        if isinstance(kwargs[kw], tuple):
            re_kwargs[kw], im_kwargs[kw] = kwargs[kw]
        else:
            if kw == 'label':
                label = kwargs['label']
                re_kwargs['label'] = label + ' (real)'
                im_kwargs['label'] = label + ' (imag)'

    if values.dtype == complex:
        line_re = ax.plot(coord, values.real, *re_args, **re_kwargs)
        line_im = ax.plot(coord, values.imag, *im_args, **im_kwargs)
        return line_re, line_im
    else:
        line_re = ax.plot(coord, values.real, *re_args, **re_kwargs)
        return line_re

import sys
def plot_error(ax, coord, values, coord_err, values_err, **kwargs):
    """keywords arguments include `ls`, `lw`, `color`, `c` or `alpha`."""
    ### TODO: correct labels
    if 'c' in kwargs:
        kwargs['color'] = kwargs.pop('c')

    line_kwargs = kwargs.copy()
    fill_kwargs = kwargs.copy()

    if 'ls' in kwargs:
        del fill_kwargs['ls']
    if 'alpha' in kwargs:
        del line_kwargs['alpha']
    else:
        fill_kwargs['alpha'] = 0.3 # default
    if 'edgecolor' in kwargs:
        del line_kwargs['edgecolor']
    else:
        fill_kwargs['edgecolor'] = ''
    if 'lw' in kwargs:
        del fill_kwargs['lw']
    fill_kwargs['lw'] = 0. # always
    if 'label' in kwargs:
        del fill_kwargs['label']

    line = ax.plot(coord, values, **line_kwargs)

    big_float = sys.float_info.max ** 0.5 # keep some margin for the plotting internal machinery
    if isinstance(values_err, tuple):
        err_lo, err_up = values_err
        err_lo, err_up = err_lo.copy(), err_up.copy()
        err_lo[np.isposinf(err_lo)] = big_float
        err_up[np.isposinf(err_up)] = big_float
        err_lo = np.interp(coord, coord_err, err_lo)
        err_up = np.interp(coord, coord_err, err_up)
        fill = ax.fill_between(coord, values - err_lo, values + err_up, **fill_kwargs)

    else:
        err = values_err.copy()
        err[np.isposinf(err)] = big_float
        err = np.interp(coord, coord_err, err)
        fill = ax.fill_between(coord, values - err, values + err, **fill_kwargs)

    return line, fill

def cpx_plot_error(ax, coord, values, coord_err, values_err, **kwargs):
    """keywords arguments include `ls`, `color` or `alpha`."""

    re_kwargs = kwargs.copy()
    im_kwargs = kwargs.copy()

    for kw in kwargs:
        if isinstance(kwargs[kw], tuple):
            re_kwargs[kw], im_kwargs[kw] = kwargs[kw]

    if isinstance(values_err, tuple):
        re_errors, im_errors = values_err
        line_re, fill_re = plot_error(ax, coord, values.real, coord_err, re_errors, **re_kwargs)
        line_im, fill_im = plot_error(ax, coord, values.imag, coord_err, im_errors, **im_kwargs)
    else:
        line_re, fill_re = plot_error(ax, coord, values.real, coord_err, values_err, **re_kwargs)
        line_im, fill_im = plot_error(ax, coord, values.imag, coord_err, values_err, **im_kwargs)

    return line_re, fill_re, line_im, fill_im

def gaussian(coord, avg=0., std=1., normed=True):
    values = np.exp(-(coord - avg)**2 / (2. * std**2))
    if normed:
        values /= np.sqrt(2.*pi) * std
    return values

def autoscale_y(ax, lines=None, margin=0.1, logmargin=False):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    lo, hi = ax.get_xlim()

    def _get_bottom_top(line):
        xd, yd = line.get_data()
        if len(yd) != 0 and not isinstance(yd, list):
            mask = np.isfinite(xd) & np.isfinite(yd)
            xd, yd = xd[mask], yd[mask]
            if logmargin:
                mask = yd > 0.
                xd, yd = xd[mask], yd[mask]
                yd = np.log(yd)
            y_displayed = yd[((xd>=lo) & (xd<=hi))]
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed)-margin*h
            top = np.max(y_displayed)+margin*h
        else:
            bot, top = 0., 0.
        if logmargin:
            return np.exp(bot), np.exp(top)
        else:
            return bot, top

    if lines is None:
        lines = ax.get_lines()
    else:
        for line in lines:
            if line not in ax.get_lines():
                raise ValueError, 'A line provided has not been plotted in `ax`'

    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = _get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def color_list(arg, cmap_name=None, default=False):
    """
    Given a size or a list of coordinates, yields nice list of colors
    """
    try:
        size = int(arg)
    except TypeError:
        is_int = False
        size = None
    else:
        is_int = True

    if default:
        if is_int:
            ### use default pyplot colors
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            while size > len(colors):
                colors += colors
            return colors[:size]
        else:
            raise TypeError

    else:
        ### define cmap
        if cmap_name is None:
            cmap_name = 'rainbow'
        cmap = plt.get_cmap(cmap_name)

        if is_int and size <= 6:
            colors = ['b', 'g', 'y', 'orange', 'r', 'm']
            return colors[:size]
        elif is_int and size > 6:
            colors = [cmap(x) for x in np.linspace(0, 1, size)]
            return colors
        else: # is not int
            coord = np.array(arg, dtype=float)
            coord -= min(coord)
            coord /= max(coord)
            colors = [cmap(x) for x in coord]
            return colors


######## Plotting 2D ########
import colorsys
### http://nbviewer.jupyter.org/github/empet/Math/blob/master/DomainColoring.ipynb

def hsv_to_rgb(H, S, V):
    shape = H.shape
    H_ = H.copy().flatten()
    S_ = S.copy().flatten()
    V_ = V.copy().flatten()
    R = np.empty_like(H_)
    G = np.empty_like(H_)
    B = np.empty_like(H_)
    for i in range(len(H_)):
        try:
            r, g, b = colorsys.hsv_to_rgb(H_[i], S_[i], V_[i])
        except ValueError:
            r, g, b = 255, 255, 255
        R[i], G[i], B[i] = r, g, b
    return R.reshape(shape), G.reshape(shape), B.reshape(shape)

def domain_coloring(w, s=0.9):
    """
    domain coloring with modulus track
    w the array of values
    s is the constant Saturation

    real positive numbers are in red, real negative in cyan.
    imag positive numbers are in green-yellow, imag negative in dark-blue-purple.
    """
    # computes the hue corresponding to the complex number w
    H = np.mod(np.angle(w) / (2*np.pi) + 1, 1)

    Logm = np.log2(np.absolute(w))
    Logm = np.nan_to_num(Logm)

    V = Logm - np.floor(Logm)
    S = s * np.ones_like(H, float)

    RGB = hsv_to_rgb(H, S, V**0.2) # V**0.2>V for V in[0,1];this choice  avoids too dark colors
    return np.dstack(RGB)

######## Sampled functions ########

def hcut(array, threshold_abs):
    if threshold_abs is None:
        return array.copy()
    mask = np.abs(array) >= threshold_abs
    output = array.copy()
    output[mask] = np.nan
    return output

def vcut(coord, values, left=None, right=None, axis=-1):
    """
    Return new arrays with less values
    """
    coord_out = coord
    values_out = np.swapaxes(values, 0, axis)
    if left is not None:
        left_i = np.argmin(np.abs(coord_out - left))
        coord_out = coord_out[left_i:]
        values_out = values_out[left_i:]
    if right is not None:
        right_i = np.argmin(np.abs(coord_out - right)) + 1
        coord_out = coord_out[:right_i]
        values_out = values_out[:right_i]
    return coord_out.copy(), np.swapaxes(values_out, 0, axis).copy()

def symmetrize(coord, values, center, function=None, axis=-1):
    coord_out = coord.copy()
    values_out = np.moveaxis(values.copy(), axis, -1)
    s = slice(None, None, -1)
    if function is None:
        function = lambda x : x
    if coord_out[0] < center < coord_out[-1]:
        raise ValueError, "center is within the coordinate range"
    elif center <= coord_out[0]:
        if center == coord_out[0]:
            s = slice(None, 0, -1)

        coord_out = np.concatenate((-coord_out[s] + 2*center, coord_out))
        values_out = np.concatenate((function(values_out[..., s]), values_out), axis=-1)

    elif center >= coord_out[-1]:
        if center == coord_out[-1]:
            s = slice(-2, None, -1)

        coord_out = np.concatenate((coord_out, -coord_out[s] + 2*center))
        values_out = np.concatenate((values_out, function(values_out[..., s])), axis=-1)

    return coord_out, np.moveaxis(values_out, -1, axis)

from scipy import interpolate
def interp_1D(x, xp, fp, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate'):
    """Wrapping of scipy.interpolate.interp1d, so as to use it in the same fashion as np.interp."""
    func = interpolate.interp1d(xp, fp, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    return func(x)

def cpx_interp(x, xp, fp, left=None, right=None, period=None):
    """Simple wrapping of numpy.interp for complex values"""
    r = np.interp(x, xp, fp.real, left, right, period)
    i = np.interp(x, xp, fp.imag, left, right, period)
    return r + 1j*i

def cpx_interp_1D(x, xp, fp, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate'):
    if fill_value != 'extrapolate':
        raise NotImplementedError
    r = interp_1D(x, xp, fp.real, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    i = interp_1D(x, xp, fp.imag, kind=kind, axis=axis, bounds_error=bounds_error, fill_value=fill_value)
    return r + 1j*i

def convolve(coord1, values1, coord2, values2):
    """
    I don't like this function. Probably better to use convolve_coord (in ctint_keldysh).
    """
    PRECISION = 1e-10
    N = len(values1)
    M = len(values2)
    assert len(coord1) == N
    dt = coord1[1] - coord1[0]

    # make a centered coord2 if not given
    if coord2 is None:
        coord2 = dt * np.arange((-M+1) / 2., (M+1) / 2.)

    assert len(coord2) == M
    assert abs(dt - coord2[1] + coord2[0]) < PRECISION

    values_conv = dt * np.convolve(values1, values2, 'valid')

    L = len(values_conv)
    if N <= M:
        coord_conv = coord2[(M-L) // 2:(M+L)// 2] + coord1[(M-L) // 2]
    else:
        coord_conv = coord1[(N-L) // 2:(N+L)// 2] + coord2[(N-L) // 2]

    assert len(coord_conv) == len(values_conv)
    return coord_conv, values_conv

def normed_gaussian_wind(step, std, span):
    N = int(0.5 * span / step)
    coord = np.linspace(-N*step, N*step, 2*N + 1)
    values = gaussian(coord, 0., std, True)
    return coord, values

def extend_coord(coord, c_left, c_right):
    dt = (coord[-1] - coord[0]) / (len(coord) - 1)
    new_coord = coord.copy()
    if c_right > coord[-1]:
        new_coord = np.concatenate((new_coord, np.arange(new_coord[-1]+dt, c_right+dt, dt)))
    if c_left < coord[0]:
        new_coord = np.concatenate((np.arange(new_coord[0]-dt, c_left-dt, -dt)[::-1], new_coord))
    return new_coord

from scipy import signal
def flying_avg(coord, values, span=None, std=None):
    dt = coord[1] - coord[0]
    if span is None:
        span = (coord[-1] - coord[0]) * 0.1
    if std is None:
        std = span * 0.25
    n = int(span / dt + 0.5)
    std_norm = std / dt
    g = signal.gaussian(n, std_norm)
    g /= dt * g.sum()
    return convolve(coord, values, None, g)

def flying_stat(coord, values, span=None, std=None):
    out_coord, avg = flying_avg(coord, values, span, std)
    out_coord, sqr_avg = flying_avg(coord, np.abs(values)**2, span, std)

    return out_coord, avg, np.sqrt(sqr_avg - np.abs(avg)**2)

from scipy.stats import mstats
def filter_outliers(values, size_box=None, mode='reflect'):
    counter = 0
    output = values.copy()
    if size_box is None:
        size_box = len(values) // 100
    i_middle = size_box // 2
    padded_values = np.pad(values, i_middle, mode)
    for i in range(len(values)):
        box = padded_values[i : i + size_box + 1]

        alpha, _, _, _ = mstats.theilslopes(box) # more robust than least square
        flat_box = box - alpha * np.arange(len(box))

        med = np.median(flat_box)
        mad = np.median(np.abs(flat_box - med)) # Median Absolute Deviation, more robust than std
        true_med = med + alpha*i_middle

        if abs(values[i] - true_med) > 10 * mad:
            output[i] = true_med
            counter += 1
    print "Values removed:", counter
    return output

def filter_cpx_outliers(values, size_box=None, mode='reflect'):
    output_r = filter_outliers(values.real, size_box, mode)
    output_i = filter_outliers(values.imag, size_box, mode)
    return output_r + 1j*output_i

######## Sampled functions : Fourier transform ########
from numpy import fft

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

def fourier_transform(times, function, n=None, axis=-1, conv=-1):
    r"""
    $f(\nu) = \int{dt} f(t) e^{-conv 2i\pi \nu t}$
    times is assumed sorted and regularly spaced
    conv = 1 or -1 allows to change convention. Default is -1.
    """

    if (len(times) != function.shape[axis]):
        raise ValueError, "`times` must have the same length as `function` on specified `axis`."
    if (conv not in [-1, +1]):
        raise ValueError, "`conv` must be -1 or +1."
    if n is None:
        n = len(times)
    elif n == 'auto':
        n = _next_regular(len(times))

    delta_t = times[1] - times[0]

    freq = fft.fftshift(fft.fftfreq(n, delta_t))
    spectrum = fft.fftshift(fft.fft(function, n=n, axis=axis), axes=axis)
    spectrum = np.swapaxes(spectrum, -1, axis)

    freq = conv * freq[::conv]
    spectrum = spectrum[..., ::conv]
    spectrum[..., :] *= delta_t * np.exp(-conv * 1j * 2*pi * freq * times[0])

    return freq, np.swapaxes(spectrum, -1, axis)

def inv_fourier_transform(freq, spectrum, n=None, axis=-1, conv=-1):
    r"""
    $f(t) = \int{d\nu} f(\nu) e^{conv 2i\pi \nu t}$
    freq is assumed sorted and regularly spaced
    conv = 1 or -1 allows to change convention. Default is -1.
    """

    if (len(freq) != spectrum.shape[axis]):
        raise ValueError, "`freq` must have the same length as `spectrum` on specified `axis`."
    if (conv not in [-1, +1]):
        raise ValueError, "`conv` must be -1 or +1."
    if n is None:
        n = len(freq)
    elif n == 'auto':
        n = _next_regular(len(freq))

    delta_f = freq[1] - freq[0]

    times = fft.fftshift(fft.fftfreq(n, delta_f))
    function = fft.fftshift(fft.ifft(spectrum, n=n, axis=axis), axes=axis)
    function = np.swapaxes(function, -1, axis)

    times = conv * times[::conv]
    function = function[..., ::conv]
    function[..., :] *= delta_f * np.exp(conv * 1j * 2*pi * times * freq[0]) * n

    return times, np.swapaxes(function, -1, axis)

def smooth_FT(times, function, side, n=None, conv=-1):
    assert side in ['left', 'right']
    if side == 'left':
        freq, spectrum = smooth_FT(times[::-1], function[..., ::-1], 'right', n, conv)
        return freq[::-1], spectrum[..., ::-1]

    # side = 'right'
    a, b = times[0], times[-1]
    dt = np.abs(times[-1] - times[-2])
    assert a != b
    fa, fb = function[..., 0], function[..., -1]
    print "fb=", fb

    k_max = 1. / dt
    k_min = np.maximum(np.log(np.abs(fb / fa)), 1.) / (b - a)
    assert (k_min < k_max).all() # TODO
    print "security factor:", k_max / k_min
    k = np.sqrt(k_min*k_max) # geometric average
    print "1/k =", 1. / k
    fb = fb[..., np.newaxis]
    k = k[..., np.newaxis]
    deltaF = fb * np.exp(k * (times - b))
    print deltaF.shape

    freq, spectrum = fourier_transform(times, function - deltaF, n, conv=conv)
    puls = -conv * 2.j * np.pi * freq
    return freq, spectrum + fb * (np.exp(b*puls) - np.exp(a*puls - k*(b-a))) / (k + puls)

def heaviside_FT(times, function, k_left, k_right, n=None, conv=-1, v_left=None, v_right=None, plot=False):
    PRECISION = 1e-5
    if abs(k_left) < PRECISION:
        k_left = 0.
    if abs(k_right) < PRECISION:
        k_right = 0.
    assert abs(k_left + k_right) > PRECISION
    if v_left is None:
        v_left = function[0]
    if v_right is None:
        v_right = function[-1]

    a = times[0]
    b = times[-1]
    ul = np.exp(k_left * (a - b))
    ur = np.exp(k_right * (a - b))
    A = (v_left - ur * v_right) / (1 - ul * ur)
    B = (v_right - ul * v_left) / (1 - ul * ur)
    DF = A * np.exp(k_left * (a - times)) + B * np.exp(k_right * (times - b))

    slope_left = -k_left*A + k_right*ur*B
    slope_right = -k_left*ul*A + k_right*B
    print '1 / k_left_eff =', abs(v_left / slope_left)
    print '1 / k_right_eff =', abs(v_right / slope_right)

    if plot:
        cpx_plot(plt, times, DF)
        plt.show()

    freq, spec = fourier_transform(times, function - DF, n, conv=conv)

    DF_freq = np.zeros(freq.shape, dtype=complex)
    ipuls = conv * 2j*pi * freq
    if k_left == 0.:
        DF_freq += A * np.exp(-0.5*(a+b) * ipuls) * (b-a) * np.sinc((b-a) * freq)
    else:
        DF_freq += A * (np.exp(-np.sign(k_left)*a*ipuls) - ul * np.exp(-np.sign(k_left)*b*ipuls)) / (k_left + ipuls)
    if k_right == 0.:
        DF_freq += B * np.exp(-0.5*(a+b) * ipuls) * (b-a) * np.sinc((b-a) * freq)
    else:
        DF_freq += B * (np.exp(-np.sign(k_right)*b*ipuls) - ur * np.exp(-np.sign(k_right)*a*ipuls)) / (k_right - ipuls)

    spec += DF_freq
    return freq, spec

def heaviside_FT_vec(times, function, k_left, k_right, n=None, conv=-1, v_left=None, v_right=None, plot=False):
    function = np.array(function)
    assert len(times) == function.shape[-1]
    shape = function.shape[:-1]

    function = function.reshape(-1, function.shape[-1])

    v_left = np.resize(v_left, function.shape[0])
    v_right = np.resize(v_right, function.shape[0])

    spectra = []
    for i in range(function.shape[0]):
        freq, spectrum = heaviside_FT(times, function[i, :], k_left, k_right, n, conv, v_left[i], v_right[i], plot)
        spectra.append(spectrum)
    spectra = np.array(spectra).reshape(shape + (len(freq),))

    return freq, spectra

######## Saving arrays in txt ########

def save_1Darrays_txt(filename, header, *args):
    data = list(args)
    np.savetxt(filename, np.vstack(data).T, header=header)

if __name__ == '__main__':

    ### test fourier transforms
    if False:
        ### gaussian
        t_ref = np.linspace(-10., 10., 500)
        gt_ref = gaussian(t_ref, 2., 1.)

        f, gf = fourier_transform(t_ref, gt_ref, n=10*len(t_ref))

        plt.plot(f, gf.real, 'b')
        plt.plot(f, gf.imag, 'r')
        plt.title('TF of a gaussian')
        plt.show()

        t, gt = inv_fourier_transform(f, gf)

        plt.plot(t_ref, gt_ref, 'k')
        plt.plot(t, gt.real, 'b')
        plt.xlim([-10, 10])
        plt.title('gaussian real part')
        plt.show()
        plt.plot(t, gt.imag, 'r')
        plt.title('gaussian imag part (=0)')
        plt.show()

        ### door
        t_ref = np.linspace(-10., 10., 500)
        gt_ref = np.zeros((len(t_ref),))
        gt_ref[np.logical_and(-5. < t_ref, t_ref < 7.)] = 1.

        f, gf = fourier_transform(t_ref, gt_ref, n=10*len(t_ref))
        plt.plot(f, gf.real, 'b')
        plt.plot(f, gf.imag, 'r')
        plt.title('TF of a door')
        plt.show()
        t, gt = inv_fourier_transform(f, gf)

        plt.plot(t_ref, gt_ref, 'k')
        plt.plot(t, gt.real, 'b')
        plt.title('door real part')
        plt.show()
        plt.plot(t, gt.imag, 'r')
        plt.title('door imag part (=0)')
        plt.show()

        ### hermitian gaussian
        t_ref = np.linspace(-10., 10., 500)
        gt_ref = gaussian(t_ref, 0., 1.) + 1j * (gaussian(t_ref, 2., 1.) - gaussian(t_ref, -2., 1.))

        f, gf = fourier_transform(t_ref, gt_ref, n=10*len(t_ref))

        plt.plot(f, gf.real, 'b')
        plt.plot(f, gf.imag, 'r')
        plt.title('TF of an hermitian gaussian (is real)')
        plt.show()

        t, gt = inv_fourier_transform(f, gf)

        plt.plot(t_ref, gt_ref.real, 'k--')
        plt.plot(t, gt.real, 'b')
        plt.xlim([-10, 10])
        plt.title('hermitian gaussian real part')
        plt.show()
        plt.plot(t_ref, gt_ref.imag, 'k--')
        plt.plot(t, gt.imag, 'r')
        plt.xlim([-10, 10])
        plt.title('hermitian gaussian imag part')
        plt.show()

        ### test FT conventions
        t = np.linspace(0, 10, 1000)
        func = np.exp(-3. * t)
        f, spec = fourier_transform(t, func, n=1000, conv=+1)
        assert (f == sorted(f)).all()
        spec_ref = 1. / (3. + 2.j * pi * f)

        plt.plot(f, spec.real, 'b')
        plt.plot(f, spec.imag, 'r')
        plt.plot(f, spec_ref.real, '--b')
        plt.plot(f, spec_ref.imag, '--r')
        plt.xlim([-5, 5])
        plt.title('TF conventions')
        plt.show()


    if False:
        ### test smooth_FT
        t = np.linspace(-10, 2, 2000)
        func = np.exp(3. * (t-2.)) + 2j*np.exp(2. * (t-2.))
        cpx_plot(plt, t, func)
        plt.show()

        f, spec = smooth_FT(t, func, 'right', n=4000)
        print spec.shape
        f, spec_raw = fourier_transform(t, func, n=4000)
        spec *= np.exp(-2.j*np.pi*f * 2.)
        spec_raw *= np.exp(-2.j*np.pi*f * 2.)
        spec_ref = 1. / (3. + 2.j*np.pi * f) + 2.j / (2. + 2.j*np.pi * f)

        cpx_plot(plt, f, spec_raw)
        plt.title('raw')
        # plt.xlim([0, 1])
        plt.show()
        cpx_plot(plt, f, spec)
        plt.title('theta')
        # plt.xlim([0, 1])
        plt.show()
        cpx_plot(plt, f, spec_ref)
        plt.title('ref')
        # plt.xlim([0, 1])
        plt.show()

        cpx_plot(plt, f, spec_raw - spec_ref)
        plt.plot(f, np.abs(spec_raw - spec_ref), 'k')
        plt.title('raw - ref')
        plt.show()
        cpx_plot(plt, f, spec - spec_ref)
        plt.plot(f, np.abs(spec - spec_ref), 'k')
        plt.title('theta - ref')
        plt.show()
        cpx_plot(plt, f, spec_raw - spec)
        plt.plot(f, np.abs(spec_raw - spec), 'k')
        plt.title('raw - theta')
        plt.show()

    ### test color_list
    coord = np.r_[0.01, 0.1, 0.5, 1., 2., 20.]
    coord_ref = coord.copy()
    c = color_list(coord)

    assert (coord == coord_ref).all()

