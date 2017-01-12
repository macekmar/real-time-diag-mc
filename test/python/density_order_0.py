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
p["measure_times"] = ([tmax], tmax)
p["weight_time"] = tmax
p["U"] = 2.5 # U_qmc
p["max_perturbation_order"] = 0
p["min_perturbation_order"] = 0
p["alpha"] = 0.5
p["n_warmup_cycles"] = 1000
p["length_cycle"] = 10
p["n_cycles"] = 5000
p["p_dbl"] = 0

(pn, sn), (pn_error, sn_error) = S.solve(**p)

g = pn[0] * sn[0]
g_ref = 6.3715406319870704E-01-7.4590093081033579E-12j

if abs(g - g_ref) > 1.e-6:
    raise RuntimeError, "FAILED"

