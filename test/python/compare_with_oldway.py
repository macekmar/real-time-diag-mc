import numpy as np
from ctint_keldysh import make_g0_semi_circular, compute_correlator_oldway, compute_correlator, make_g0_contour
from pytriqs.archive import HDFArchive
from matplotlib import pyplot as plt

from toolbox import cpx_plot_error

filename_kernel = 'out_files/impurity.out.h5'
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

### load oldway
times_oldway = np.linspace(-40.0, 0.0, 101)[10::10]
error_o = lambda a : np.sqrt(np.var(a, ddof=1, axis=-1) / float(a.shape[-1]))

with HDFArchive(filename_oldway, 'r') as ar:

    LGF, LGF_part = [], []
    GGF, GGF_part = [], []
    for k, t in enumerate(times_oldway):
        lgf, lgf_part = compute_correlator_oldway(ar['less_{0}'.format(k)], np.abs(g0_less_triqs[0, 0](t)))
        LGF.append(lgf)
        LGF_part.append(lgf_part)

        ggf, ggf_part = compute_correlator_oldway(ar['grea_{0}'.format(k)], np.abs(g0_grea_triqs[0, 0](t)))
        GGF.append(ggf)
        GGF_part.append(ggf_part)

    LGF = np.array(LGF)
    LGF_re_err = error_o(np.real(LGF_part))
    LGF_im_err = error_o(np.imag(LGF_part))
    GGF = np.array(GGF)
    GGF_re_err = error_o(np.real(GGF_part))
    GGF_im_err = error_o(np.imag(GGF_part))


### load kernel
g0_contour = make_g0_contour(g0_less_triqs[0, 0], g0_grea_triqs[0, 0])

with HDFArchive(filename_kernel, 'r') as ar:
    times, GF, times_part, GF_part = compute_correlator(ar['run_1'], g0_contour)

    nbs_k = ar['run_1']['parameters']['nb_bins_sum']
error_k = lambda a: np.sqrt(nbs_k * np.var(a, ddof=1, axis=-1) / float(a.shape[-1]))

### remove orbital axis
GF = GF[:, :, :, 0]
GF_part = GF_part[:, :, :, 0]

### compute error
GF_re_err = error_k(GF_part.real)
GF_im_err = error_k(GF_part.imag)




### order 1

fig, ax = plt.subplots(2, 1, figsize=(20, 16))
with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    ax[0].plot(ref['less']['times'], ref['less']['o1'].real, 'g.')
    ax[0].plot(ref['less']['times'], ref['less']['o1'].imag, 'm.')
    cpx_plot_error(ax[0], times, GF[0, :, 0], times_part,
                   (GF_re_err[0, :, 0], GF_im_err[0, :, 0]), c=('b', 'r'))
    ax[0].errorbar(times_oldway, LGF[:, 1].real, LGF_re_err[:, 1], fmt='bo')
    ax[0].errorbar(times_oldway, LGF[:, 1].imag, LGF_re_err[:, 1], fmt='ro')
    ax[0].set_xlim(-42, 2)
    ax[0].set_title('lesser n=1')

    ax[1].plot(ref['grea']['times'], ref['grea']['o1'].real, 'g.')
    ax[1].plot(ref['grea']['times'], ref['grea']['o1'].imag, 'm.')
    cpx_plot_error(ax[1], times, GF[0, :, 1], times_part,
                   (GF_re_err[0, :, 1], GF_im_err[0, :, 1]), c=('b', 'r'))
    ax[1].errorbar(times_oldway, GGF[:, 1].real, GGF_re_err[:, 1], fmt='bo')
    ax[1].errorbar(times_oldway, GGF[:, 1].imag, GGF_re_err[:, 1], fmt='ro')
    ax[1].set_xlim(-42, 2)
    ax[1].set_title('greater n=1')
plt.show()

### order 2

fig, ax = plt.subplots(2, 1, figsize=(20, 16))
with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
    ax[0].errorbar(ref['less']['times'], ref['less']['o2'].real, ref['less']['o2_error'].real, fmt='g.')
    ax[0].errorbar(ref['less']['times'], ref['less']['o2'].imag, ref['less']['o2_error'].imag, fmt='m.')
    cpx_plot_error(ax[0], times, GF[1, :, 0], times_part,
                   (GF_re_err[1, :, 0], GF_im_err[1, :, 0]), c=('b', 'r'))
    ax[0].errorbar(times_oldway, LGF[:, 2].real, LGF_re_err[:, 2], fmt='bo')
    ax[0].errorbar(times_oldway, LGF[:, 2].imag, LGF_re_err[:, 2], fmt='ro')
    ax[0].set_xlim(-42, 2)
    ax[0].set_title('lesser n=2')

    ax[1].errorbar(ref['grea']['times'], ref['grea']['o2'].real, ref['grea']['o2_error'].real, fmt='g.')
    ax[1].errorbar(ref['grea']['times'], ref['grea']['o2'].imag, ref['grea']['o2_error'].imag, fmt='m.')
    cpx_plot_error(ax[1], times, GF[1, :, 1], times_part,
                   (GF_re_err[1, :, 1], GF_im_err[1, :, 1]), c=('b', 'r'))
    ax[1].errorbar(times_oldway, GGF[:, 2].real, GGF_re_err[:, 2], fmt='bo')
    ax[1].errorbar(times_oldway, GGF[:, 2].imag, GGF_re_err[:, 2], fmt='ro')
    ax[1].set_xlim(-42, 2)
    ax[1].set_title('greater n=2')
plt.show()
