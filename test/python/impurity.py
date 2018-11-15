from pytriqs.utility import mpi
from ctint_keldysh import make_g0_semi_circular, solve, compute_correlator, make_g0_contour
from pytriqs.archive import HDFArchive
import numpy as np
import os
from matplotlib import pyplot as plt
import warnings
from copy import deepcopy

from toolbox import cpx_plot_error

"""
This tests the python interface of the solver in the staircase usage, and the accuracy of the calculation at orders 1 and 2. It computes an impurity problem with no orbitals.

Takes about 5 min for 10000 cycles
"""

if mpi.world.size < 2:
    warnings.warn('This test is run on a single process. It is advised to run it on several processes to carry a more thorough test.', RuntimeWarning)

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)

filename = 'out_files/' + os.path.basename(__file__)[:-3] + '.out.h5'
p = {}

p["staircase"] = True
p["nb_warmup_cycles"] = 1000
p["nb_cycles"] = 10000#0
p["save_period"] = 60*60
p["filename"] = filename
p["run_name"] = 'run_1'
p["g0_lesser"] = g0_less_triqs[0, 0]
p["g0_greater"] = g0_grea_triqs[0, 0]
p["size_part"] = 10
p["nb_bins_sum"] = 10

### computes G^<(t) and G^\bar{T}(t) = G^>(t) for t<0
p["creation_ops"] = [(0, 0, 0.0, 1)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False
p["interaction_start"] = 40.0
p["alpha"] = 0.0
p["nb_orbitals"] = 1
p["potential"] = ([1.], [0], [0])

p["U"] = 4.5 # U_qmc
p["w_ins_rem"] = 0.5
p["w_dbl"] = 0.
p["w_shift"] = 1.
p["max_perturbation_order"] = 2
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 50
p["method"] = 1
p["singular_thresholds"] = [3.5, 3.3]

p_copy = deepcopy(p) # keep parameters safe

solve(**p)

if mpi.world.rank == 0:

    error = lambda a: np.sqrt(p_copy["nb_bins_sum"] * np.var(a, ddof=1, axis=-1) / float(a.shape[-1]))
    g0_contour = make_g0_contour(g0_less_triqs[0, 0], g0_grea_triqs[0, 0])

    with HDFArchive(filename, 'r') as ar:
        times, GF, times_part, GF_part = compute_correlator(ar['run_1'], g0_contour)

    ### remove orbital axis
    GF = GF[:, :, :, 0]
    GF_part = GF_part[:, :, :, 0]

    ### compute error
    GF_re_err = error(GF_part.real)
    GF_im_err = error(GF_part.imag)

    ### order 1

    fig, ax = plt.subplots(2, 2)
    with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
        cpx_plot_error(ax[0, 0], times, GF[0, :, 0], times_part,
                       (GF_re_err[0, :, 0], GF_im_err[0, :, 0]), c=('b', 'r'))
        ax[0, 0].plot(ref['less']['times'], ref['less']['o1'].real, 'g.-')
        ax[0, 0].plot(ref['less']['times'], ref['less']['o1'].imag, 'm.-')
        ax[0, 0].set_title('less o1')

        cpx_plot_error(ax[0, 1], times, GF[0, :, 1], times_part,
                       (GF_re_err[0, :, 1], GF_im_err[0, :, 1]), c=('b', 'r'))
        ax[0, 1].plot(ref['grea']['times'], ref['grea']['o1'].real, 'g.-')
        ax[0, 1].plot(ref['grea']['times'], ref['grea']['o1'].imag, 'm.-')
        ax[0, 1].set_title('grea o1')

    ### order 2

    with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
        cpx_plot_error(ax[1, 0], times, GF[1, :, 0], times_part,
                       (GF_re_err[1, :, 0], GF_im_err[1, :, 0]), c=('b', 'r'))
        ax[1, 0].errorbar(ref['less']['times'], ref['less']['o2'].real,
                       ref['less']['o2_error'].real, fmt='g.-')
        ax[1, 0].errorbar(ref['less']['times'], ref['less']['o2'].imag,
                       ref['less']['o2_error'].imag, fmt='m.-')
        ax[1, 0].set_title('less o2')

        cpx_plot_error(ax[1, 1], times, GF[1, :, 1], times_part,
                       (GF_re_err[1, :, 1], GF_im_err[1, :, 1]), c=('b', 'r'))
        ax[1, 1].errorbar(ref['grea']['times'], ref['grea']['o2'].real,
                       ref['grea']['o2_error'].real, fmt='g.-')
        ax[1, 1].errorbar(ref['grea']['times'], ref['grea']['o2'].imag,
                       ref['grea']['o2_error'].imag, fmt='m.-')
        ax[1, 1].set_title('grea o2')

    plt.show()

