import numpy as np
from ctint_keldysh import make_g0_semi_circular
from pytriqs.archive import HDFArchive
from ctint_keldysh.construct_gf import kernel_GF_from_archive, oldway_GF_from_archive
from matplotlib import pyplot as plt

from toolbox import cpx_plot_error

filename_kernel = 'out_files/order1_params1_staircase.out.h5'
# filename_oldway = 'out_files/impurity_oldway.old.out.h5'
filename_oldway = 'out_files/impurity_oldway.out.h5'


p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)

g0_less = np.vectorize(lambda t: g0_less_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])
g0_grea = np.vectorize(lambda t: g0_grea_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])
g0_reta = np.vectorize(lambda t: 0. if (t >= 100. or t < 0.) else g0_grea_triqs(t)[0, 0] - g0_less_triqs(t)[0, 0], otypes=[complex])
g0_adva = np.vectorize(lambda t: 0. if (t <= -100. or t > 0.) else g0_less_triqs(t)[0, 0] - g0_grea_triqs(t)[0, 0], otypes=[complex])

error = lambda a : np.sqrt(10 * np.var(a, axis=-1) / float(a.shape[-1]))

# with HDFArchive(filename_kernel, 'r') as ar:
#     kernel = kernel_GF_from_archive(ar['run_1']['results_all'], ar['run_1']['parameters']['nonfixed_op'])
#     kernel_part = kernel_GF_from_archive(ar['run_1']['results_part'], ar['run_1']['parameters']['nonfixed_op'])

with HDFArchive(filename_kernel, 'r') as ar:
    kernel = kernel_GF_from_archive(ar['kernels'], False)
    kernel_part = kernel_GF_from_archive(ar['kernels_part'], False)

GF = kernel.convol_g_left(g0_less, g0_grea, 100.)
GF.apply_gf_sym()
GF.increment_order(g0_less, g0_grea)

GF_part = kernel_part.convol_g_left(g0_less, g0_grea, 100.)
GF_part.apply_gf_sym()
GF_part.increment_order(g0_less, g0_grea)

with HDFArchive(filename_oldway, 'r') as ar:

    times_oldway = np.linspace(-40, 0, 10)
    ar_list = [ar['less_{0}'.format(k)]['results_all'] for k in range(len(times_oldway))]
    LGF = oldway_GF_from_archive(ar_list, times_oldway, g0_less_triqs[0, 0], 'negative')
    ar_list = [ar['less_{0}'.format(k)]['results_part'] for k in range(len(times_oldway))]
    LGF_part = oldway_GF_from_archive(ar_list, times_oldway, g0_less_triqs[0, 0], 'negative')

    ar_list = [ar['grea_{0}'.format(k)]['results_all'] for k in range(len(times_oldway))]
    GGF = oldway_GF_from_archive(ar_list, times_oldway, g0_grea_triqs[0, 0], 'negative')
    ar_list = [ar['grea_{0}'.format(k)]['results_part'] for k in range(len(times_oldway))]
    GGF_part = oldway_GF_from_archive(ar_list, times_oldway, g0_grea_triqs[0, 0], 'negative')

### order 1

fig, ax = plt.subplots(2, 1)
with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    ax[0].plot(ref['less']['times'], ref['less']['o1'].real, 'g.')
    ax[0].plot(ref['less']['times'], ref['less']['o1'].imag, 'm.')
    cpx_plot_error(ax[0], GF.times, GF.less.values[1], GF_part.times,
                   (error(GF_part.less.values[1].real), error(GF_part.less.values[1].imag)),
                   color=('b', 'r'))
    ax[0].errorbar(LGF.times, LGF.values[1].real, error(LGF_part.values[1].real), fmt='bo')
    ax[0].errorbar(LGF.times, LGF.values[1].imag, error(LGF_part.values[1].imag), fmt='ro')
    ax[0].set_xlim(-50, 10)

    ax[1].plot(ref['grea']['times'], ref['grea']['o1'].real, 'g.')
    ax[1].plot(ref['grea']['times'], ref['grea']['o1'].imag, 'm.')
    cpx_plot_error(ax[1], GF.times, GF.grea.values[1], GF_part.times,
                   (error(GF_part.grea.values[1].real), error(GF_part.grea.values[1].imag)),
                   color=('b', 'r'))
    ax[1].errorbar(GGF.times, GGF.values[1].real, error(GGF_part.values[1].real), fmt='bo')
    ax[1].errorbar(GGF.times, GGF.values[1].imag, error(GGF_part.values[1].imag), fmt='ro')
    ax[1].set_xlim(-50, 10)
plt.show()

### order 2

fig, ax = plt.subplots(2, 1)
with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
    ax[0].errorbar(ref['less']['times'], ref['less']['o2'].real, ref['less']['o2_error'].real, fmt='g.')
    ax[0].errorbar(ref['less']['times'], ref['less']['o2'].imag, ref['less']['o2_error'].imag, fmt='m.')
    cpx_plot_error(ax[0], GF.times, GF.less.values[2], GF_part.times,
                   (error(GF_part.less.values[2].real), error(GF_part.less.values[2].imag)),
                   color=('b', 'r'))
    ax[0].errorbar(LGF.times, LGF.values[2].real, error(LGF_part.values[2].real), fmt='bo')
    ax[0].errorbar(LGF.times, LGF.values[2].imag, error(LGF_part.values[2].imag), fmt='ro')
    ax[0].set_xlim(-50, 10)

    ax[1].errorbar(ref['grea']['times'], ref['grea']['o2'].real, ref['grea']['o2_error'].real, fmt='g.')
    ax[1].errorbar(ref['grea']['times'], ref['grea']['o2'].imag, ref['grea']['o2_error'].imag, fmt='m.')
    cpx_plot_error(ax[1], GF.times, GF.grea.values[2], GF_part.times,
                   (error(GF_part.grea.values[2].real), error(GF_part.grea.values[2].imag)),
                   color=('b', 'r'))
    ax[1].errorbar(GGF.times, GGF.values[2].real, error(GGF_part.values[2].real), fmt='bo')
    ax[1].errorbar(GGF.times, GGF.values[2].imag, error(GGF_part.values[2].imag), fmt='ro')
    ax[1].set_xlim(-50, 10)
plt.show()
