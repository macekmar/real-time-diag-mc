# Imports
import ctint_keldysh as ctk
import ctint_keldysh.quasimc_utility as ctku
import ctint_keldysh.quasimc as ctkq
from ctint_keldysh.quasimc_post_treatment import *
from ctint_keldysh.generators import *


import numpy as np


# Parameters
p_green = {'Gamma': 0.2,
           'Nt_gf0': 50000,
           'beta': 10000.0,
           'epsilon_d': 0.2,
           'muL': 0.0,
           'muR': 0.0,
           'tmax_gf0': 200.0}

# Physical parameters
p_cpp = {'interaction_start': 100.0,
         'alpha': 0.5}
     
p_cpp['annihilation_ops'] = [[0, 0, 0.0, 0]]
p_cpp['creation_ops'] = [[0, 0, 0.0, 0]]
p_cpp['nb_orbitals'] = 1
p_cpp['potential'] = [[1.0], [0], [0]]
p_cpp['extern_alphas'] = [0.0]
p_cpp['method'] = 0
p_cpp['nb_bins'] = 10000
p_cpp['nonfixed_op'] = False
p_cpp['singular_thresholds'] = [3.5, 3.3]
p_cpp['sampling_model_intervals'] =  [[]]
p_cpp['sampling_model_coeff'] =  [[[]]]
p_cpp['max_perturbation_order'] = 6
p_cpp['U'] = [1.0 for i in range(p_cpp['max_perturbation_order'])]
params_all = dict(p_cpp)

# Set non-interacting Green's function
g0_less, g0_grea = ctk.make_g0_flat_band(**p_green)
params_all['g0_lesser'] = g0_less[0,0]
params_all['g0_greater'] = g0_grea[0,0]
S = ctk.SolverCore(**p_cpp)
S.set_g0(params_all['g0_lesser'], params_all['g0_greater'])
S.init_measure(1)

# Model/warping
def fun(V):
    return [np.abs(S.evaluate_qmc_weight([(0,0,float(-x)) for x in v])) for v in V]

# We only need one function, others will be the same
integral, inv_cdf = ctku.calculate_inv_cdfs([fun], (0.0,100.0,1001))
# Add v_i, i > 1
integral = integral*6
inv_cdf = inv_cdf*6
# Calculate nD integrals
model_ints = [np.prod(integral[:i+1]) for i in range(len(integral))]
model_ints = np.insert(model_ints, 0, 1) # 0-th order
# Set model
intervals = [inv_cdf[i].x.tolist() for i in range(params_all['max_perturbation_order'])]
coeff = [inv_cdf[i].c.T.tolist() for i in range(params_all['max_perturbation_order'])]
S.set_model(intervals, coeff)

# Test u to v to u
order = 5
for diag in range(1, order + 1):
    u_lin = np.linspace(-100, 0, order)[:,np.newaxis]
    u0 = np.zeros_like(u_lin)
    
    U = np.hstack((np.repeat(u_lin, diag, axis=1), np.repeat(u0, order-diag, axis=1)))
    V = np.array([S.u_to_v([x for x in u]) for u in U])
    W = np.array([S.v_to_u([x for x in u]) for u in V])

    V2 = V
    V2[:, diag-1] = 0.0

    assert np.allclose(V2, np.zeros_like(V2))
    assert np.allclose(U,W)

# Test u to l to u:
N_pts = 6
order = 4
U = -100.0*np.random.rand(N_pts,order) # U doesn't have to be ordered ...
L = np.array([S.u_to_l([x for x in u]) for u in U])
W = np.array([S.l_to_u([x for x in u]) for u in L])
U = np.sort(U, axis=1) # ... but it have to be for comparison with W
assert np.allclose(U,W)

print 'SUCCESS!'
