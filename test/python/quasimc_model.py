import numpy as np
import scipy.integrate as scintg
import scipy.interpolate as scintp

import ctint_keldysh as ctk
import ctint_keldysh.quasimc as ctkq
from ctint_keldysh.quasimc_utility import calculate_inv_cdfs

# This test checks if the interpolation in the C++ code returns the same results
# as the Python code.
# Slight differences are irrelevant (~1e-16), since all the calculations are 
# done with the c++ code, python should only be used to obtain the coefficients.

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
         'max_perturbation_order': 10,
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
         'w_shift': 0,
         'sampling_model_intervals': [[]],
         'sampling_model_coeff': [[[]]]}

p_py = {'nb_bins_sum': 1,
        'run_name': 'Run',
        'save_period': 10000,
        'size_part': 1,
        'staircase': True}

p_cpp['U'] = [p_cpp['U']]*p_cpp['max_perturbation_order']


# We have to reinitilaize the solver core. Previously max order was 20 so.
# The solver will generate unnecessary binning  up to this order, and we will
# have to calculate cn up to this order and thus provide p_py['model']
# This can be cleaner in the script for computations
g0_less, g0_grea = ctk.make_g0_flat_band(**p_green)
S = ctk.SolverCore(**p_cpp)
S.set_g0(g0_less[0,0], g0_grea[0,0])

def fun(u):
    return [np.abs(S.evaluate_qmc_weight([(0,0,float(x)) for x in c])) for c in u]
model_funs = [fun for i in range(p_cpp['max_perturbation_order'])]
integral, inv_cdf = calculate_inv_cdfs(model_funs, t_min=-100, Nt=1001)

# Set coefficients for solver
intervals = [inv_cdf[i].x.tolist() for i in range(p_cpp['max_perturbation_order'])]
coeff = [inv_cdf[i].c.T.tolist() for i in range(p_cpp['max_perturbation_order'])]
S.set_model(intervals, coeff)



for order in range(p_cpp['max_perturbation_order']+1):
    for itr in range(10): # Try 10 times
        r = np.random.rand(order)
        py_res = 1
        for i,x in enumerate(r):
            dp = inv_cdf[i].derivative()
            py_res /= dp(x)
        cpp_res = S.evaluate_model(r.tolist())

        assert np.allclose(py_res, cpp_res, atol=1e-16, rtol=1e-16)

print 'SUCCESS!'