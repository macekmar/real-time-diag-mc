from mpi4py import MPI
import ctint_keldysh as ctk
import ctint_keldysh.quasimc as ctkq
from ctint_keldysh.generators import *
import numpy as np
import sys

# Part of run_quasimc_diff_part.sh

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
         'extern_alphas': [0.0],
         'forbid_parity_order': -1,
         'interaction_start': 100.0,
         'length_cycle': 1,
         'max_perturbation_order': 20,
         'method': 0,
         'min_perturbation_order': 0,
         'nb_bins': 10000,
         'nb_orbitals': 1,
         'nonfixed_op': False,
         'potential': [[1.0], [0], [0]],
         'singular_thresholds': [3.5, 3.3],
         'store_configurations': 0,
         'w_dbl': 0,
         'w_ins_rem': 1.0,
         'w_shift': 0,
         'sampling_model_intervals': [[]],
         'sampling_model_coeff': [[[]]]}

p_py = {'nb_bins_sum': 1,
        'run_name': 'Run',
        'save_period': 37,
        'size_part': 1,
        'staircase': True}

p_cpp['U'] = [p_cpp['U']]*p_cpp['max_perturbation_order']

# Merge and set parameters
params_all = dict(p_cpp, **p_py)

N_vec = [   np.linspace(0, 100, 11, dtype=np.int).tolist(),
            np.linspace(0, 200, 11, dtype=np.int).tolist(),
            np.linspace(0, 300, 11, dtype=np.int).tolist()]

params_all['N'] = N_vec

params_all['order'] = [2,3,4]
params_all['max_perturbation_order'] = max(params_all['order'])
params_all['U'] = params_all['max_perturbation_order']*[1.0]

# Irrelevant, but necessary
params_all['nb_warmup_cycles'] = 1
params_all["nb_cycles"] = 10000
# 
params_all['filename'] = "Test" + str(MPI.COMM_WORLD.size) + ".hdf5"
params_all['num_gen'] = SobolGenerator

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
S.init_measure(1)

# Asymmetric model
def fun(u):
    return [np.abs(S.evaluate_qmc_weight([(0,0,float(x)) for x in c])) for c in u]
params_all['model'] = [fun for i in range(params_all['max_perturbation_order'])]

p = ctkq.quasi_solver(S, **params_all) 
