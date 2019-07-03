from mpi4py import MPI
import ctint_keldysh as ctk
import ctint_keldysh.quasimc as ctkq
from ctint_keldysh.generators import *
from pytriqs.archive import HDFArchive
from ctint_keldysh.quasimc_post_treatment import *
import numpy as np
import sys

# Part of run_quasimc_diff_part.sh

# Parameters
p_green = {'Gamma': 0.2,
           'epsilon_d': 0.2,
           'beta': 10000.0,
           'muL': 0.0,
           'muR': 0.0,
           'Nt_gf0': 50000,
           'tmax_gf0': 1000.0}

p_cpp = {'interaction_start': 100.0,
         'alpha': 0.5}    
p_cpp['annihilation_ops'] = [[0, 0, 0.0, 0]]
p_cpp['creation_ops'] = [[0, 0, 0.0, 0]]
p_cpp['nb_orbitals'] = 1
p_cpp['potential'] = [[1.0], [0], [0]]
p_cpp['extern_alphas'] = [0.0]
p_cpp['method'] = 1
p_cpp['nb_bins'] = 10000
p_cpp['singular_thresholds'] = [3.5, 3.3]
p_cpp['sampling_model_intervals'] =  [[]]
p_cpp['sampling_model_coeff'] =  [[[]]]

p_py = {'run_name': 'Run',
        'filename': "Test" + str(MPI.COMM_WORLD.size) + ".hdf5",
        'order': [2,3,4],
        'num_gen': SobolGenerator,
        'num_gen_seed': 1,
        'keep_cubic_domain': False,
        'save_period': 37}

N_vec = [   np.linspace(0, 3000, 11, dtype=np.int).tolist(),
            np.linspace(0, 200, 11, dtype=np.int).tolist(),
            np.linspace(0, 300, 11, dtype=np.int).tolist()]
p_py['N'] = N_vec

# Merge and set parameters
p_cpp['max_perturbation_order'] = max(p_py['order'])
p_cpp['U'] = [1.0]*p_cpp['max_perturbation_order']
params_all = dict(p_cpp, **p_py)


# We have to reinitilaize the solver core. Previously max order was 20 so.
# The solver will generate unnecessary binning  up to this order, and we will
# have to calculate cn up to this order and thus provide p_py['model']
# This can be cleaner in the script for computations
g0_less, g0_grea = ctk.make_g0_flat_band(**p_green)
params_all['g0_lesser'] = g0_less[0,0]
params_all['g0_greater'] = g0_grea[0,0]

S = ctk.SolverCore(**p_cpp)
S.set_g0(params_all['g0_lesser'], params_all['g0_greater'])
S.init_measure(1)

# Asymmetric model
def fun(V):
    return [np.abs(S.evaluate_qmc_weight([(0,0,float(-x)) for x in c])) for c in V]
params_all['model'] = [fun for i in range(params_all['max_perturbation_order'])]

p = ctkq.quasi_solver(S, **params_all) 


# Check accuracy for order 2
with HDFArchive("Test" + str(MPI.COMM_WORLD.size) + ".hdf5", "r") as f:  
    run = p_py["run_name"] 
    # Because of nb_bins_sum we have to keep _final and _inter separate.
    # Again, _final is not really necessary.
    bin_times_final = f[run]["results_final"]["bin_times"]
    kernels_final = f[run]["results_final"]["kernels"] # Indices: requested orders, bin, keldysh index, orbital 
    bin_times_inter = f[run]["results_inter"]["bin_times"]
    kernels_inter = f[run]["results_inter"]["kernels"] # Indices: requested orders, bin, keldysh index, orbital, N_samples
    weight = f[run]["results_inter"]["weight"] 
    abs_weight = f[run]["results_inter"]["abs_weight"] 
    orders = f[run]["metadata"]["orders"]
    N_gen = f[run]['results_inter']['N_generated'] # Indices are: order, N_samples
    N_calc = f[run]['results_inter']['N_calculated']

# Calculate G(w) ... (w is omega)
w_window = (-2,2)
w_num_points = 2001

w, Gw = get_GR_w(bin_times_inter, kernels_inter, orders, w_window, w_num_points, p_green['epsilon_d'], p_green['Gamma'])

ind = np.argmin(np.abs(w)) # point closest to 0
true_val = 0.01563883-0.00562187j # Taken from Corentin's data
error = np.abs(Gw[0,ind,0,-1] - true_val)/np.abs(true_val) 
assert error < 1e-2

if MPI.COMM_WORLD.rank == 0:
        print "SUCCESS!"