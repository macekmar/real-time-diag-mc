from mpi4py import MPI
from ctint_keldysh import SolverCore, make_g0_semi_circular
import numpy as np

"""
Test collecting in oldway method

"""

world = MPI.COMM_WORLD
if world.size != 4:
    raise RuntimeError, 'This test has to be run on 4 processes to be meaningful.'

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)
g0_less = g0_less_triqs[0, 0]
g0_grea = g0_grea_triqs[0, 0]


p = {}
p["creation_ops"] = [(0, 0, 0.0, 1)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False # annihilation
p["interaction_start"] = 40.0
p["alpha"] = 0.5
p["nb_orbitals"] = 1
p["potential"] = ([1.], [0], [0])

p["U"] = 3. # U_qmc
p["w_ins_rem"] = 1
p["w_dbl"] = 0
p["w_shift"] = 0
p["max_perturbation_order"] = 4
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 10
p["verbosity"] = 0
p["method"] = 0
p["singular_thresholds"] = [3.0, 3.3]

S = SolverCore(**p)
S.set_g0(g0_less, g0_grea)

S.run(50, False) # warmup
S.run(50, True)
pn_each = S.pn.copy()
sn_each = S.sn.copy()

is_group_master = S.collect_results(2) # collect in two groups
group = world.Split(is_group_master) # creates 2 communicators, one for the masters and one for the others. Only the masters' matters
pn_part = S.pn.copy()
sn_part = S.sn.copy()

S.collect_results(1) # collect all
pn_all = S.pn.copy()
sn_all = S.sn.copy()

try:
    ### test that each, part and all are different
    assert (pn_each != pn_all).any()
    assert (pn_part != pn_all).any()
    assert (pn_each != pn_part).any()

    assert (sn_each != sn_all).any()
    assert (sn_part != sn_all).any()
    assert (sn_each != sn_part).any()

    ### reduction of `each` or `part` should give `all`
    if world.rank == 0:
        print '\n'
        print 'pn_all:', pn_all
        print 'sn_all:', sn_all

    pn_each_reduced = world.reduce(pn_each)
    sn_each_reduced = world.reduce(pn_each * sn_each)
    if is_group_master:
        pn_part_reduced = group.reduce(pn_part)
        sn_part_reduced = group.reduce(pn_part * sn_part)
    if world.rank == 0:
        assert (pn_each_reduced == pn_all).all()
        assert (pn_part_reduced == pn_all).all() # note that rank=0 should be a master

        assert np.allclose(sn_each_reduced, pn_all * sn_all)
        assert np.allclose(sn_part_reduced, pn_all * sn_all)

    ### test re-collecting
    S.collect_results(4) # individual data
    assert (S.pn == pn_each).all()
    assert (S.sn == sn_each).all()
    S.collect_results(2) # two groups
    assert (S.pn == pn_part).all()
    assert (S.sn == sn_part).all()

except AssertionError:
    world.Abort(1)

if world.rank == 0:
    print 'SUCCESS !'

