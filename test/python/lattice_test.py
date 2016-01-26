import pytriqs.utility.mpi as MPI
from pytriqs.applications.impurity_solvers.ctint_new import *

# constructor parameters
d1 = {}
d1["beta"] = 10.
d1["mu"] = 0.5
d1["L"] = 50
d1["n_freq"] = 500
d1["t_min"] = -30.0
d1["t_max"] = 30.0

# construct solver
S = CtintSolver(**d1)
print "non-interacting charge: ", S.c0

# solve parameters
d2 = {}
d2["U"] = 1.0
d2["L"] = 50
d2["max_perturbation_order"] = 4
d2["tmax"] = 10.0
d2["alpha"] = 0.0
d2["n_cycles"] = 10000
d2["n_warmup_cycles"] = 100
d2["length_cycle"]= 10
d2["p_dbl"] = -1

# construct solver
S.solve(**d2)

print S.CnSn
