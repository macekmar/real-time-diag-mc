from mpi4py import MPI
import ctint_keldysh as ctk
import ctint_keldysh.model as ctkm
import ctint_keldysh.quasimc as ctkq
import ctint_keldysh.gen_harmonic as harmonic
import matplotlib.pyplot as plt
import numpy as np
from pytriqs.archive import HDFArchive
import sys

N_samples = int(sys.argv[1])

# Parameters
p_green = {'Gamma': 0.2,
           'Nt_gf0': 5000,
           'beta': 10000.0,
           'epsilon_d': 0.2,
           'muL': 0.0,
           'muR': 0.0,
           'tmax_gf0': 100}

p_cpp = {'U': 1.0,
         'alpha': 0.5,
         'annihilation_ops': [[0, 0, 0.0, 0]],
         'creation_ops': [[0, 0, 0.0, 0]],
         'do_quasi_mc': True,
         'extern_alphas': [0.0],
         'forbid_parity_order': -1,
         'interaction_start': 100.0,
         'length_cycle': 1,
         'max_perturbation_order': 20,
         'method': 1,
         'min_perturbation_order': 0,
         'nb_bins': 50000,
         'nb_orbitals': 1,
         'nonfixed_op': False,
         'potential': [[1.0], [0], [0]],
         'singular_thresholds': [3.5, 3.3],
         'store_configurations': 0,
         'w_dbl': 0,
         'w_ins_rem': 1.0,
         'w_shift': 0}

p_py = {'nb_bins_sum': 1,
        'run_name': 'Run',
        'save_period': 10000,
        'size_part': 1,
        'staircase': True}

p_cpp['U'] = [p_cpp['U']]*p_cpp['max_perturbation_order']

# Merge and set parameters
params_all = dict(p_cpp, **p_py)
params_all['N'] = [N_samples]
params_all['order'] = [2]
params_all['max_perturbation_order'] = max(params_all['order'])
params_all['U'] = params_all['max_perturbation_order']*[1.0]

# Irrelevant, but necessary
params_all['nb_warmup_cycles'] = 1
params_all["nb_cycles"] = 10000
# 
params_all['filename'] = "Test" + str(MPI.COMM_WORLD.size) + ".hdf5"
params_all['num_gen'] = harmonic.HarmonicGenerator

# We have to reinitilaize the solver core. Previously max order was 20 so.
# The solver will generate unnecessary binning  up to this order, and we will
# have to calculate cn up to this order and thus provide p_py['model']
# This can be cleaner in the script for computations
p_cpp_temp = dict([(key, params_all[key]) for key in p_cpp.keys()]) 
g0_less, g0_grea = ctk.make_g0_flat_band(**p_green)
params_all['g0_lesser'] = g0_less[0,0]
params_all['g0_greater'] = g0_grea[0,0]
S = ctk.SolverCore(**p_cpp_temp)
S.set_g0(params_all['g0_lesser'], params_all['g0_greater'])

# Asymmetric model
def fun(u, do_measure=False):
    return ctkm.get(S, u, do_measure)
params_all['model'] = [fun for i in range(params_all['max_perturbation_order'])]

p = ctkq.quasi_solver(S, **params_all) 
