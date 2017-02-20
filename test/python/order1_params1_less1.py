from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import os

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_lesser, g0_greater = make_g0_semi_circular(**p)

times = np.linspace(-40.0, 0.0, 101)
p = {}
p["right_input_points"] = [(0, 0.0, 1)] # lesser
p["interaction_start"] = 40.0
p["measure_state"] = 0
p["measure_keldysh_indices"] = [0] # lesser
p["measure_times"] = times
p["U"] = 2.5 # U_qmc
p["min_perturbation_order"] = 0
p["max_perturbation_order"] = 1
p["alpha"] = 0.0
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 50
p["n_cycles"] = 100000
p["w_dbl"] = 0
p["w_shift"] = 0
p["w_weight_swap"] = 0.2
p["method"] = 4

on, on_error = single_solve(g0_lesser, g0_greater, p)

if mpi.is_master_node():
    with HDFArchive('out_files/' + os.path.basename(__file__)[:-3] + '.out.h5', 'w') as ar:  # A file to store the results
        ar.create_group('less')
        less = ar['less']
        less['on'] = on
        less['times'] = times

with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ar:
    if not np.array_equal(times, ar['less']['times']):
        raise RuntimeError, 'FAILED: times are different'

    if not np.allclose(on[1], ar['less']['o1'], rtol=0.1, atol=0.01):
        raise RuntimeError, 'FAILED'

