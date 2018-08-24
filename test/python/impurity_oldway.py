from pytriqs.utility import mpi
from ctint_keldysh import make_g0_semi_circular, solve
from ctint_keldysh.construct_gf import oldway_GF_from_archive
from pytriqs.archive import HDFArchive
import numpy as np
import os
from matplotlib import pyplot as plt
import warnings
from copy import deepcopy
from datetime import datetime

"""
This tests the python interface of the solver in the staircase usage and oldway algorithm (determinant), and the accuracy of the calculation at orders 1 and 2. It computes an impurity problem with no orbitals.
"""

if mpi.world.size < 2:
    warnings.warn('This test is run on a single process. It is advised to run it on several processes to carry a more thorough test.', RuntimeWarning)

start_time = datetime.now()

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)

filename = 'out_files/' + os.path.basename(__file__)[:-3] + '.out.h5'

p = {}

p["staircase"] = True
p["nb_warmup_cycles"] = 1000
p["nb_cycles"] = 1000#0#0
p["save_period"] = 10*60
p["filename"] = filename
p["run_name"] = 'run'
p["g0_lesser"] = g0_less_triqs[0, 0]
p["g0_greater"] = g0_grea_triqs[0, 0]
p["size_part"] = 10
p["nb_bins_sum"] = 10

p["creation_ops"] = [(0, 0, 0.0, 0)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False
p["interaction_start"] = 40.0
p["alpha"] = 0.0
p["nb_orbitals"] = 1
p["potential"] = ([1.], [0], [0])

p["U"] = 4.5 # U_qmc
p["w_ins_rem"] = 0.5
p["w_dbl"] = 0.
p["w_shift"] = 1.
p["max_perturbation_order"] = 2
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 50
p["method"] = 0 # oldway
p["singular_thresholds"] = [3.5, 3.3]

p_copy = deepcopy(p) # keep parameters safe

times = np.linspace(-40, 0, 10)

### compute lesser
for k, t in enumerate(times):
    if mpi.world.rank == 0:
        print '################# Lesser ###################'
        print '                 {0} / {1}'.format(k+1, len(times))
    p["run_name"] = 'less_{0}'.format(k)
    p["annihilation_ops"] = [(0, 0, t, 0)]
    p["creation_ops"] = [(0, 0, 0.0, 1)]
    # solve(p)

### compute greater
for k, t in enumerate(times):
    if mpi.world.rank == 0:
        print '################# Greater ###################'
        print '                 {0} / {1}'.format(k+1, len(times))
    p["run_name"] = 'grea_{0}'.format(k)
    p["annihilation_ops"] = [(0, 0, t, 1)]
    p["creation_ops"] = [(0, 0, 0.0, 0)]
    # solve(p)

if mpi.world.rank == 0:

    with HDFArchive(filename, 'r') as ar:

        LGF = oldway_GF_from_archive([ar['less_{0}'.format(k)]['results_all'] for k in range(len(times))],
                                     times, g0_less_triqs[0, 0], 'negative')

        LGF_part = oldway_GF_from_archive([ar['less_{0}'.format(k)]['results_part'] for k in range(len(times))],
                                     times, g0_less_triqs[0, 0], 'negative')

        GGF = oldway_GF_from_archive([ar['grea_{0}'.format(k)]['results_all'] for k in range(len(times))],
                                     times, g0_grea_triqs[0, 0], 'negative')

        GGF_part = oldway_GF_from_archive([ar['grea_{0}'.format(k)]['results_part'] for k in range(len(times))],
                                     times, g0_grea_triqs[0, 0], 'negative')

    error = lambda a : np.sqrt(p_copy['nb_bins_sum'] * np.var(a, axis=-1) / float(a.shape[-1]))

    ### order 1

    fig, ax = plt.subplots(2, 2)
    with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
        ax[0, 0].errorbar(times, LGF.values[1].real, error(LGF_part.values[1].real), fmt='bo')
        ax[0, 0].plot(ref['less']['times'], ref['less']['o1'].real, 'g.-')
        ax[0, 0].errorbar(times, LGF.values[1].imag, error(LGF_part.values[1].imag), fmt='ro')
        ax[0, 0].plot(ref['less']['times'], ref['less']['o1'].imag, 'm.-')
        ax[0, 0].set_title('less o1')

        ax[0, 1].errorbar(times, GGF.values[1].real, error(GGF_part.values[1].real), fmt='bo')
        ax[0, 1].plot(ref['grea']['times'], ref['grea']['o1'].real, 'g.-')
        ax[0, 1].errorbar(times, GGF.values[1].imag, error(GGF_part.values[1].imag), fmt='ro')
        ax[0, 1].plot(ref['grea']['times'], ref['grea']['o1'].imag, 'm.-')
        ax[0, 1].set_title('grea o1')

    ### order 2

    with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
        ax[1, 0].errorbar(times, LGF.values[2].real, error(LGF_part.values[2].real), fmt='bo')
        ax[1, 0].errorbar(ref['less']['times'], ref['less']['o2'].real, ref['less']['o2_error'].real, fmt='g.-')
        ax[1, 0].errorbar(times, LGF.values[2].imag, error(LGF_part.values[2].imag), fmt='ro')
        ax[1, 0].errorbar(ref['less']['times'], ref['less']['o2'].imag, ref['less']['o2_error'].imag, fmt='m.-')
        ax[1, 0].set_title('less o2')

        ax[1, 1].errorbar(times, GGF.values[2].real, error(GGF_part.values[2].real), fmt='bo')
        ax[1, 1].errorbar(ref['grea']['times'], ref['grea']['o2'].real, ref['grea']['o2_error'].real, fmt='g.-')
        ax[1, 1].errorbar(times, GGF.values[2].imag, error(GGF_part.values[2].imag), fmt='ro')
        ax[1, 1].errorbar(ref['grea']['times'], ref['grea']['o2'].imag, ref['grea']['o2_error'].imag, fmt='m.-')
        ax[1, 1].set_title('grea o2')

    plt.show()

    print 'Run time =', str(datetime.now() - start_time)

