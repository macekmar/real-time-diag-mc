from pytriqs.utility import mpi
from ctint_keldysh import make_g0_semi_circular, solve
from ctint_keldysh.construct_gf import kernel_GF_from_archive
from pytriqs.archive import HDFArchive
import numpy as np
import os
from matplotlib import pyplot as plt
import warnings
from copy import deepcopy

"""
This tests the python interface of the solver in the staircase usage, and the accuracy of the calculation at orders 1 and 2. It computes an impurity problem with no orbitals.
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
p["save_period"] = 3*60
p["filename"] = filename
p["run_name"] = 'run_1'
p["g0_lesser"] = g0_less_triqs[0, 0]
p["g0_greater"] = g0_grea_triqs[0, 0]
p["size_part"] = 10
p["nb_bins_sum"] = 10

p["creation_ops"] = [(0, 0, 0.0, 0)]
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

solve(p)

if mpi.world.rank == 0:

    with HDFArchive(filename, 'r') as ar:

        ### check nb_kernels agrees with pn
        res = ar['run_1']['results_all']
        if (res['nb_kernels'][0, :, 0].sum(axis=0) != res['pn'][1] or
            res['nb_kernels'][0, :, 1].sum(axis=0) != res['pn'][1]):
            raise RuntimeError
        if (res['nb_kernels'][1, :, 0].sum(axis=0) != 2*res['pn'][2] or
            res['nb_kernels'][1, :, 1].sum(axis=0) != 2*res['pn'][2]):
            raise RuntimeError
        res = ar['run_1']['results_part']
        if mpi.world.size >= 2:
            if ((res['nb_kernels'][0, :, 0].sum(axis=0) != res['pn'][1]).any() or
                (res['nb_kernels'][0, :, 1].sum(axis=0) != res['pn'][1]).any()):
                raise RuntimeError
            if ((res['nb_kernels'][1, :, 0].sum(axis=0) != 2*res['pn'][2]).any() or
                (res['nb_kernels'][1, :, 1].sum(axis=0) != 2*res['pn'][2]).any()):
                raise RuntimeError
        else:
            if (res['nb_kernels'][0, :, 0].sum(axis=0) != res['pn'][1] or
                res['nb_kernels'][0, :, 1].sum(axis=0) != res['pn'][1]):
                raise RuntimeError
            if (res['nb_kernels'][1, :, 0].sum(axis=0) != 2*res['pn'][2] or
                res['nb_kernels'][1, :, 1].sum(axis=0) != 2*res['pn'][2]):
                raise RuntimeError

        kernel = kernel_GF_from_archive(ar['run_1']['results_all'], p_copy["nonfixed_op"])
        kernel_part = kernel_GF_from_archive(ar['run_1']['results_part'], p_copy["nonfixed_op"])


        if False: # plot kernel order 1
            res = ar['run_1']['results_all']
            error = lambda a : np.sqrt(np.var(a, axis=-1) / float(a.shape[-1]))
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].plot(kernel.times, kernel.less.values[0].real, '.b', markersize=1)
            ax[0, 0].plot(kernel.times, kernel.less.values[0].imag, '.r', markersize=1)
            ax[0, 0].set_xlim(-50, 10)
            # ax[1, 0].plot(kernel_part.times, error(kernel_part.less.values[0].real), 'b')
            # ax[1, 0].plot(kernel_part.times, error(kernel_part.less.values[0].imag), 'r')
            ax[1, 0].plot(res['bin_times'], res['nb_kernels'][0, :, 0], 'k')
            ax[1, 0].set_xlim(-50, 10)
            ax[0, 1].plot(kernel.times, kernel.grea.values[0].real, '.b', markersize=1)
            ax[0, 1].plot(kernel.times, kernel.grea.values[0].imag, '.r', markersize=1)
            ax[0, 1].set_xlim(-50, 10)
            # ax[1, 1].plot(kernel_part.times, error(kernel_part.grea.values[0].real), 'b')
            # ax[1, 1].plot(kernel_part.times, error(kernel_part.grea.values[0].imag), 'r')
            ax[1, 1].plot(res['bin_times'], res['nb_kernels'][0, :, 1], 'k')
            ax[1, 1].set_xlim(-50, 10)
            plt.tight_layout()
            plt.show()

        if False: # plot kernel order 2
            error = lambda a : np.sqrt(np.var(a, axis=-1) / float(a.shape[-1]))
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].plot(kernel.times, kernel.less.values[1].real, '.b', markersize=1)
            ax[0, 0].plot(kernel.times, kernel.less.values[1].imag, '.r', markersize=1)
            ax[0, 0].set_xlim(-50, 10)
            ax[1, 0].plot(kernel_part.times, error(kernel_part.less.values[1].real), 'b')
            ax[1, 0].plot(kernel_part.times, error(kernel_part.less.values[1].imag), 'r')
            ax[1, 0].set_xlim(-50, 10)
            ax[0, 1].plot(kernel.times, kernel.grea.values[1].real, '.b', markersize=1)
            ax[0, 1].plot(kernel.times, kernel.grea.values[1].imag, '.r', markersize=1)
            ax[0, 1].set_xlim(-50, 10)
            ax[1, 1].plot(kernel_part.times, error(kernel_part.grea.values[1].real), 'b')
            ax[1, 1].plot(kernel_part.times, error(kernel_part.grea.values[1].imag), 'r')
            ax[1, 1].set_xlim(-50, 10)
            plt.show()

    g0_less = np.vectorize(lambda t: g0_less_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])
    g0_grea = np.vectorize(lambda t: g0_grea_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])
    g0_reta = np.vectorize(lambda t: 0. if (t >= 100. or t < 0.) else g0_grea_triqs(t)[0, 0] - g0_less_triqs(t)[0, 0], otypes=[complex])
    g0_adva = np.vectorize(lambda t: 0. if (t <= -100. or t > 0.) else g0_less_triqs(t)[0, 0] - g0_grea_triqs(t)[0, 0], otypes=[complex])

    exit()
    ############# lesser and greater ##############
    GF = kernel.convol_g_left(g0_less, g0_grea, 100.)
    GF.apply_gf_sym()
    GF.increment_order(g0_less, g0_grea)

    if GF.less.values.ndim != 2 or GF.grea.values.ndim != 2:
        raise RuntimeError, 'FAILED'

    ### test calculation of error (not compared to anything)
    GF_part = kernel_part.convol_g_left(g0_less, g0_grea, 100.)
    GF_part.apply_gf_sym()
    GF_part.increment_order(g0_less, g0_grea)
    error = lambda a : np.sqrt(np.var(a, axis=-1) / float(a.shape[-1]))

    if mpi.world.size < 2:
        if GF_part.less.values.ndim != 2 or GF_part.grea.values.ndim != 2:
            raise RuntimeError, 'FAILED'
    else:
        if GF_part.less.values.ndim != 3 or GF_part.grea.values.ndim != 3:
            raise RuntimeError, 'FAILED'
        if error(GF_part.less.values).ndim != 2 or error(GF_part.grea.values).ndim != 2:
            raise RuntimeError, 'FAILED'
        nb_part = min(mpi.world.size, p_copy["size_part"])
        if GF_part.less.values.shape[-1] != nb_part or GF_part.grea.values.shape[-1] != nb_part:
            raise RuntimeError, 'FAILED'

    ### test number of measures
    with HDFArchive(filename, 'r') as ar:
        if not np.array_equal(ar['run_1']['metadata']['nb_measures'], mpi.world.size * p_copy["nb_cycles"] * np.ones(2)):
            raise RuntimeError, 'FAILED'


    ### test order 0

    if False:
        plt.plot(GF.times, GF.less.values[0].real, 'b')
        plt.plot(GF.times, GF.less.values[0].imag, 'r')
        plt.plot(GF.times, g0_less(GF.times).real, 'g')
        plt.plot(GF.times, g0_less(GF.times).imag, 'm')
        plt.show()

    rtol = 0.001
    atol = 0.0001
    if not np.allclose(GF.less.values[0], g0_less(GF.times), rtol=rtol, atol=atol):
        raise RuntimeError, 'FAILED o0 less'

    if not np.allclose(GF.grea.values[0], g0_grea(GF.times), rtol=rtol, atol=atol):
        raise RuntimeError, 'FAILED o0 grea'


    print 'SUCCESS !'
