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

S = SolverCore(g0_lesser, g0_greater)

times = np.linspace(-40.0, 0.0, 101)
times = times[::10] # divide by 10 the size because oldway takes more time
p = {}
p["op_to_measure"] = [[(0, 1), (0, 0)], []] # greater
p["interaction_start"] = 40.0
p["U"] = 2.5 # U_qmc
p["min_perturbation_order"] = 0
p["alpha"] = 0.0
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 50
p["n_cycles"] = 100000
p["p_dbl"] = 0
p["p_shift"] = 0
p["p_weight_time_swap"] = 0.2
p["method"] = 0

on = np.empty((2, len(times)), dtype=complex)

for i, t in enumerate(times):
    p["measure_times"] = [t]

    p["max_perturbation_order"] = 0
    (p0, s0), _ = S.solve(**p)
    c0 = abs(p0 * s0)

    p["max_perturbation_order"] = 1
    (pn, sn), _ = S.solve(**p)

    on[:, i] = perturbation_series(c0, pn, sn, p["U"])[:, 0]

if mpi.is_master_node():
    with HDFArchive('out_files/' + os.path.basename(__file__)[:-3] + '.out.h5', 'a') as ar:  # A file to store the results
        ar['on_grea'] = on
        ar['times'] = times

with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ar:
    if not np.array_equal(times, ar['grea']['times'][::10]):
        raise RuntimeError, 'FAILED: times are different'

    if not np.allclose(on[1], ar['grea']['o1'][::10], rtol=0.01, atol=0.005):
        raise RuntimeError, 'FAILED'

