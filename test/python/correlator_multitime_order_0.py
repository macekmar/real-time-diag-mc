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
p["measure_times"] = ([0.9 * t1, t1, 1.1 * t1], t1)
p["weight_times"] = t1, t2
p["U"] = 2.5 # U_qmc
p["max_perturbation_order"] = 0
p["min_perturbation_order"] = 0
p["alpha"] = 0.0
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 10
p["n_cycles"] = 5000
p["p_dbl"] = 0

(pn, sn), (pn_error, sn_error) = S.solve(**p)

g = pn[0] * sn[0, 1]
g_ref = 4.9796814938486443E-02-2.3135219474631394E-01j

if abs(g - g_ref) > 1.e-6:
    raise RuntimeError, "FAILED"

