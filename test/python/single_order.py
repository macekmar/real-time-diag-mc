from pytriqs.utility import mpi
from ctint_keldysh import make_g0_semi_circular, solve
from ctint_keldysh.construct_gf import kernel_GF_from_archive
from pytriqs.archive import HDFArchive
import numpy as np
import os
import warnings
from copy import deepcopy

"""
This tests the python interface of the solver for a single pertubation order
(ie parameter `staircase` set to False). It does not compare to any reference
data. The test succeed if the code run without trouble.
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

p["staircase"] = False
p["nb_warmup_cycles"] = 10
p["nb_cycles"] = 100
p["save_period"] = 10
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

p["U"] = 2.5 # U_qmc
p["w_ins_rem"] = 1.
p["w_dbl"] = 0
p["w_shift"] = 0
p["max_perturbation_order"] = 2
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 50
p["method"] = 1
p["singular_thresholds"] = [3.5, 3.3]

p_copy = deepcopy(p) # keep parameters safe

results = solve(p)

if mpi.world.rank == 0:

    with HDFArchive(filename, 'r') as ar:
        kernel = kernel_GF_from_archive(ar['run_1']['results_all'], p_copy["nonfixed_op"])
        kernel_part = kernel_GF_from_archive(ar['run_1']['results_part'], p_copy["nonfixed_op"])

    g0_less = np.vectorize(lambda t: g0_less_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])
    g0_grea = np.vectorize(lambda t: g0_grea_triqs(t)[0, 0] if np.abs(t) < 100. else 0., otypes=[complex])

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
        if not np.array_equal(ar['run_1']['metadata']['nb_measures'], mpi.world.size * p_copy["nb_cycles"] * np.ones(1)):
            raise RuntimeError, 'FAILED'


    print 'SUCCESS !'
