from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Test the symmetry of the kernels in the particle-hole symmetric case (epsilon_d=0, alpha=0.5)

"""

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.2
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.0
p["muL"] = 0.
p["muR"] = 0.

g0_lesser, g0_greater = make_g0_semi_circular(**p)

# enforce the particle-hole symmetry
# g0_greater.data[...] = np.conjugate(g0_lesser.data)

less_data = g0_lesser.data[:, 0, 0]
g0_lesser.data[:, 0, 0]  = 0.5*(less_data - np.conjugate(less_data)[::-1])

grea_data = g0_greater.data[:, 0, 0]
g0_greater.data[:, 0, 0]  = 0.5*(grea_data - np.conjugate(grea_data)[::-1])

times = np.linspace(-40.0, 0.0, 101)
p = {}
p["creation_ops"] = [(0, 0.0, 0)]
p["annihilation_ops"] = []
p["interaction_start"] = 40.0
p["measure_state"] = 0
p["measure_keldysh_indices"] = [0, 1]
p["measure_times"] = times
p["U"] = 0.5 # U_qmc
p["min_perturbation_order"] = 0
p["max_perturbation_order"] = 4
p["alpha"] = 0.5
p["length_cycle"] = 50
p["w_ins_rem"] = 0
p["w_dbl"] = 1
p["w_shift"] = 0
p["method"] = 5
p["singular_thresholds"] = [3.0, 3.3]

S = SolverCore(**p)
S.set_g0(g0_lesser, g0_greater)

S.run(1000, False) # warmup
S.run(10000, True)

kernels = S.kernels
kernels_all = S.kernels_all

for k in range(len(kernels)):

    # plt.plot(np.abs(kernels[k, :, 0] - np.conjugate(kernels[k, :, 1])))
    # plt.show()
    if not np.allclose(kernels[k, :, 0], np.conjugate(kernels[k, :, 1]), atol=1e-12):
        raise RuntimeError, 'FAILED kernels not symmetric for k={0}'.format(k)
    if not np.allclose(kernels_all[k, :, 0], np.conjugate(kernels_all[k, :, 1]), atol=1e-12):
        raise RuntimeError, 'FAILED kernels_all not symmetric for k={0}'.format(k)


print 'SUCCESS !'
