from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import time

p = {}
p["beta"] = 100.0
p["Gamma"] = 0.5*0.5 # gamma = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.
p["muL"] = 0.5
p["muR"] = 0

g0_lesser, g0_greater = make_g0_semi_circular(**p)

S = SolverCore(g0_lesser, g0_greater)
tmax = 3.

p = {}
p["op_to_measure"] = [[(0, 0), (0, 1)], []]
p["measure_times"] = ([0.5*tmax, tmax, 1.5*tmax], tmax)
p["weight_times"] = tmax, tmax
p["U"] = 2.5 # U_qmc
p["max_perturbation_order"] = 7
p["min_perturbation_order"] = 0
p["alpha"] = 0.5
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 10
p["n_cycles"] = 20000
p["p_dbl"] = 0
p["p_weight_time_swap"] = 1.0

(pn, sn), (pn_error, sn_error) = S.solve(**p)

if mpi.is_master_node():
    with HDFArchive('density20k.out.h5', 'a') as ar:  # A file to store the results
        ar['pn_values'] = pn
        ar['pn_errors'] = pn_error
        ar['sn_values'] = sn[:, 1]
        ar['sn_errors'] = sn_error

from pytriqs.utility.h5diff import h5diff

# Compare with density calculated from commit 63cfc21cf5211572c6ffbec7093654e2e219b63a
with HDFArchive('density20k.ref.h5', 'r') as ar:
    print "sn out:", sn[:, 1]
    print "sn ref:", ar['sn_values']
    print "rel diff:", np.abs((sn[:, 1] - ar['sn_values'])/ar['sn_values']).mean()
    print "pn out:", pn
    print "pn ref:", ar['pn_values']
    print "rel diff:", np.abs((pn - ar['pn_values'])/ar['pn_values']).mean()

h5diff("density20k.out.h5","density20k.ref.h5")
