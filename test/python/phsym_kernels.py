from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Test the symmetry of the kernels in the particle-hole symmetric case (epsilon_d=0, alpha=0.5)

"""

from pytriqs.gf.local import GfReTime
from scipy import special

def g0_less_ana(Gamma, t):
    """g0 lesser sans interaction pour le car particule trou symmetrique (epsilon_d=0.),
    a l'equilibre et a temperature nulle."""
    if t > 0.:
        im = 0.5 * np.exp(-Gamma*np.abs(t))
        re = np.exp(Gamma*t) * special.expi(-t*Gamma).real - np.exp(-Gamma*t) * special.expi(t*Gamma).real
        re /= 2.*np.pi
        return re + 1.j*im
    elif t < 0.:
        return -np.conjugate(g0_less_ana(Gamma, -t))
    else:
        return 0.5j

g0_less_ana_v = np.vectorize(g0_less_ana)

Nt_gf0 = 10000
tmax_gf0 = 110.
g0_less = GfReTime(indices=['d', 's'], window=(-tmax_gf0, tmax_gf0), n_points=2*Nt_gf0+1, name='lesser')
g0_grea = GfReTime(indices=['d', 's'], window=(-tmax_gf0, tmax_gf0), n_points=2*Nt_gf0+1, name='greater')

Gamma = 0.2
g0_less.data[:, 0, 0] = g0_less_ana_v(Gamma, np.linspace(-tmax_gf0, tmax_gf0, 2*Nt_gf0+1))
g0_grea.data[:, 0, 0] = np.conjugate(g0_less.data[:, 0, 0])


times = np.linspace(-40.0, 0.0, 101)
p = {}
p["creation_ops"] = [(0, 0.0, 0)]
p["annihilation_ops"] = []
p["extern_alphas"] = [0.]
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
S.set_g0(g0_less, g0_grea)

S.run(1000, False) # warmup
S.run(10000, True)

kernels = S.kernels

for k in range(1, len(kernels), 2):

    # plt.plot(np.abs(kernels[k, :, 0] - np.conjugate(kernels[k, :, 1])))
    # plt.semilogy()
    # plt.show()
    if not np.allclose(kernels[k, :, 0], np.conjugate(kernels[k, :, 1]), atol=1e-12):
        raise RuntimeError, 'FAILED kernels not symmetric for k={0}'.format(k)


print 'SUCCESS !'
