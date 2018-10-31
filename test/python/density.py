from pytriqs.utility import mpi
from ctint_keldysh import SolverCore, make_g0_semi_circular
from pytriqs.archive import HDFArchive
import numpy as np

### TODO: WIP

p = {}
p["beta"] = 100.0
p["Gamma"] = 0.5*0.5 # gamma = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.
p["muL"] = 0.5
p["muR"] = 0

g0_lesser, g0_greater = make_g0_semi_circular(**p)

tmax = 3.

p = {}
p["creation_ops"] = [(0, 0, 0.0, 1)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False
p["interaction_start"] = tmax
p["alpha"] = 0.5
p["nb_orbitals"] = 1
p["potential"] = ([1.], [0], [0])

p["U"] = [2.5] * 7 # U_qmc
p["w_ins_rem"] = 1
p["w_dbl"] = 0
p["w_shift"] = 0
p["max_perturbation_order"] = 7
p["min_perturbation_order"] = 0
p["length_cycle"] = 10
p["method"] = 0
p["singular_thresholds"] = [3.0, 3.3]

S = SolverCore(**p)
S.set_g0(g0_lesser[0, 0], g0_greater[0, 0])

S.run(1000, False) # warmup
S.run(20000, True)
S.collect_results(1)

pn = S.pn
sn = S.sn

print
print 'pn:', pn

pn_rel = np.array(pn, dtype=float) / float(S.nb_measures)
sn_pos = np.imag(sn)

if mpi.world.rank == 0:
    with HDFArchive('ref_data/density.ref.h5', 'r') as ar:
        success = True

        print
        print 'pn:    ', pn_rel
        print 'pn ref:', ar['pn_values']
        print 'pn err:', ar['pn_errors']
        if not (np.abs(pn_rel - ar['pn_values']) < 1e-10 + 2 * ar['pn_errors']).all():
            print
            print 'Fail pn'
            success = False

        print
        print 'sn:    ', sn_pos
        print 'sn ref:', ar['sn_values']
        print 'sn err:', ar['sn_errors']
        if not (np.abs(sn_pos - ar['sn_values']) < 1e-10 + 2 * ar['sn_errors']).all():
            print
            print 'Fail sn'
            success = False

    if not success:
        raise RuntimeError, 'FAILED'

    print 'SUCCESS !'

