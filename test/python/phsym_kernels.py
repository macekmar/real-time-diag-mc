from pytriqs.utility import mpi
from ctint_keldysh import SolverCore
import numpy as np
import matplotlib.pyplot as plt

"""
Test the symmetry of the kernels in the particle-hole symmetric case (epsilon_d=0, alpha=0.5)
Double moves needed as odd orders are zero.

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
g0_less = GfReTime(indices=[0], window=(-tmax_gf0, tmax_gf0), n_points=2*Nt_gf0+1, name='lesser')
g0_grea = GfReTime(indices=[0], window=(-tmax_gf0, tmax_gf0), n_points=2*Nt_gf0+1, name='greater')

Gamma = 0.2
g0_less.data[:, 0, 0] = g0_less_ana_v(Gamma, np.linspace(-tmax_gf0, tmax_gf0, 2*Nt_gf0+1))
g0_grea.data[:, 0, 0] = np.conjugate(g0_less.data[:, 0, 0])


p = {}
p["creation_ops"] = [(0, 0, 0.0, 0)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False # annihilation
p["interaction_start"] = 40.0
p["alpha"] = 0.5
p["nb_orbitals"] = 1
p["potential"] = ([1.], [0], [0])

p["U"] = 0.5 # U_qmc
p["w_ins_rem"] = 0
p["w_dbl"] = 1
p["w_shift"] = 0
p["max_perturbation_order"] = 4
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = 1
p["length_cycle"] = 50
p["verbosity"] = 1 if mpi.world.rank == 0 else 0
p["method"] = 1
p["singular_thresholds"] = [3.0, 3.3]

S = SolverCore(**p)
S.set_g0(g0_less, g0_grea)

S.run(100, False) # warmup
S.run(1000, True)
S.collect_results(1) # gather on all processes

if mpi.world.rank == 0:
    kernels = S.kernels

    nb_measures = S.nb_measures
    ref_nb_measures = 1000 * mpi.world.size
    if nb_measures != ref_nb_measures:
        raise RuntimeError, 'FAILED: Solver reported having completed {0} measures instead of {1}'.format(nb_measures, ref_nb_measures)


    for k in range(1, len(kernels), 2):

        # plt.plot(np.abs(kernels[k, :, 0] - np.conjugate(kernels[k, :, 1])))
        # plt.semilogy()
        # plt.show()
        if not np.allclose(kernels[k, :, 0], np.conjugate(kernels[k, :, 1]), atol=1e-12):
            raise RuntimeError, 'FAILED kernels not symmetric for k={0}'.format(k)


    print 'SUCCESS !'
