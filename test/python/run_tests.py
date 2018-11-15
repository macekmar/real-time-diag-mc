from mpi4py import MPI
from ctint_keldysh import make_g0_semi_circular, solve
import warnings
import os
import traceback
from copy import deepcopy
from time import sleep

"""
This tests the python interface of the solver in the staircase scheme
for different situations. It does not compare to any reference
data. The test succeed if the code runs without trouble.
"""

def print_traceback():
    if MPI.COMM_WORLD.size < 2:
        traceback.print_exc()
    else:
        print '>>> rank {0} failed, traceback:'.format(MPI.COMM_WORLD.rank)
        traceback.print_exc()
        print

class TracebackPrinter(object):

    def __init__(self):
        self.lockfile = '.run_tests.lock'
        self._is_locked = False

    def __del__(self):
        self._unlock()

    def _lock(self):
        while True:
            try:
                with open(self.lockfile, 'wx'):
                    pass
                self._is_locked = True
                break
            except IOError:
                sleep(0.5)

    def _unlock(self):
        if self._is_locked:
            os.remove(self.lockfile)
            self._is_locked = False

    def __call__(self):
        self._lock()
        print_traceback()
        self._unlock()

traceback_printer = TracebackPrinter()


if MPI.COMM_WORLD.size < 2:
    warnings.warn('This test is run on a single process. It is advised to run it on several processes to carry a more thorough test.', RuntimeWarning)

situations = []
situations.append({'method': 0, 'forbid_parity_order': -1, 'w_ins_rem': 1., 'w_dbl': 0.})
situations.append({'method': 0, 'forbid_parity_order': 1, 'w_ins_rem': 0., 'w_dbl': 1.})
situations.append({'method': 1, 'forbid_parity_order': -1, 'w_ins_rem': 1., 'w_dbl': 0.})
situations.append({'method': 1, 'forbid_parity_order': 1, 'w_ins_rem': 0., 'w_dbl': 1.})

nb_tests = len(situations)
success_list = []
for k, sit in enumerate(situations):
    if MPI.COMM_WORLD.rank == 0:
        print '>>>>>>>>> Trying situation: ', sit
        print

    try:

        p = {}
        p["beta"] = 200.
        p["Gamma"] = 0.5
        p["tmax_gf0"] = 100.
        p["Nt_gf0"] = 2500
        p["epsilon_d"] = 0.5
        p["muL"] = 0.
        p["muR"] = 0.

        g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)

        filename = 'out_files/' + os.path.basename(__file__)[:-3] + '.out.h5'
        p = {}

        p["staircase"] = True
        p["nb_warmup_cycles"] = 100
        p["nb_cycles"] = 100
        p["save_period"] = 10
        p["filename"] = filename
        p["run_name"] = 'situation_{0}'.format(k)
        p["g0_lesser"] = g0_less_triqs[0, 0]
        p["g0_greater"] = g0_grea_triqs[0, 0]
        p["size_part"] = 10
        p["nb_bins_sum"] = 10

        p["creation_ops"] = [(0, 0, 0.0, 1)]
        p["annihilation_ops"] = [(0, 0, 0.0, 0)]
        p["extern_alphas"] = [0.]
        p["nonfixed_op"] = False
        p["interaction_start"] = 40.0
        p["alpha"] = 0.0
        p["nb_orbitals"] = 1
        p["potential"] = ([1.], [0], [0])

        p["U"] = 2.5 # U_qmc
        p["w_ins_rem"] = 1.
        p["w_dbl"] = 0.
        p["w_shift"] = 0.
        p["max_perturbation_order"] = 4
        p["min_perturbation_order"] = 0
        p["forbid_parity_order"] = -1
        p["length_cycle"] = 50
        p["method"] = 0
        p["singular_thresholds"] = [3.5, 3.3]

        for key in p:
            if key in sit:
                p[key] = deepcopy(sit.pop(key))

        if len(sit) > 0:
            raise RuntimeError('Some data in `situation` has not been used')

        p_copy = deepcopy(p) # keep parameters safe

        results = solve(**p)

    except:
        success = False
        traceback_printer()

    else:
        success = True

    success_all = MPI.COMM_WORLD.allreduce(success, op=MPI.PROD)
    if MPI.COMM_WORLD.rank == 0:
        success_list.append(success_all)
        print '>>> ', 'SUCCESS' if success_all else 'FAIL'
        print

if MPI.COMM_WORLD.rank == 0:
    print '>>> Results of {0} tests:'.format(nb_tests)
    for s in success_list:
        print bool(s)
