from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import time

p = {}
p["beta"] = 100.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.
p["muL"] = 0.5
p["muR"] = 0

g0_lesser, g0_greater = make_g0_semi_circular(**p)

S = SolverCore(g0_lesser, g0_greater)
t1 = 3.
t2 = 5.

p = {}
p["op_to_measure"] = [[(0, 0), (0, 1)], []] # lesser
p["interaction_start"] = t2
p["measure_times"] = [0.9 * t1 - t2, t1 - t2]#, 1.1 * t1 - t2]
p["weight_time"] = t1 - t2
p["U"] = 2.5 # U_qmc
p["max_perturbation_order"] = 7
p["min_perturbation_order"] = 0
p["alpha"] = 0.0
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 10
p["n_cycles"] = 5000
p["p_dbl"] = 0

(pn, sn), (pn_error, sn_error) = S.solve(**p)

if mpi.is_master_node():
    with HDFArchive('correlator.out.h5', 'a') as ar:  # A file to store the results
        ar['pn_values'] = pn
        ar['pn_errors'] = pn_error
        ar['sn_values'] = sn[:, 1]
        # ar['sn_errors'] = sn_error

from pytriqs.utility.h5diff import h5diff

with HDFArchive('correlator.ref.h5', 'r') as ar:
    print "sn out:", sn[:, 1]
    print "sn ref:", ar['sn_values']
    print "rel diff:", np.abs((sn[:, 1] - ar['sn_values'])/ar['sn_values']).mean()
    print "pn out:", pn
    print "pn ref:", ar['pn_values']
    print "rel diff:", np.abs((pn - ar['pn_values'])/ar['pn_values']).mean()

h5diff("correlator.out.h5","correlator.ref.h5")
