from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import os

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_lesser, g0_greater = make_g0_semi_circular(**p)

times = np.linspace(-40.0, 0.0, 101)
p = {}
p["creation_ops"] = [(0, 0.0, 0)]
p["annihilation_ops"] = []
p["extern_alphas"] = [0.]
p["interaction_start"] = 40.0
p["measure_state"] = 0
p["measure_keldysh_indices"] = [0, 1]
p["measure_times"] = times
p["U"] = 2.5 # U_qmc
p["min_perturbation_order"] = 0
p["max_perturbation_order"] = 1
p["alpha"] = 0.0
p["length_cycle"] = 50
p["w_dbl"] = 0
p["w_shift"] = 0
p["method"] = 5
p["singular_thresholds"] = [3.5, 3.3]

S = SolverCore(**p)
S.set_g0(g0_lesser, g0_greater)

c0, _ = S.order_zero

S.run(1000, False) # warmup
S.run(100000, True)

S.compute_sn_from_kernels

on = perturbation_series(c0, S.pn, S.sn, p["U"])
on = np.squeeze(on)

on_all = perturbation_series(c0, S.pn_all, S.sn_all, p["U"])
on_all = np.squeeze(on_all)

if mpi.is_master_node():
    with HDFArchive('out_files/' + os.path.basename(__file__)[:-3] + '.out.h5', 'w') as ar:
        ar['on'] = on
        ar['on_all'] = on_all
        ar['times'] = times

if on.shape != (2, 101, 2):
    raise RuntimeError, 'FAILED: on shape is ' + str(on.shape) + ' but should be (2, 101, 2)'

if on_all.shape != (2, 101, 2):
    raise RuntimeError, 'FAILED: on_all shape is ' + str(on.shape) + ' but should be (2, 101, 2)'


# order 0
rtol = 0.001
atol = 0.0001
o0_less = np.array([g0_lesser(t)[0, 0] for t in times], dtype=complex)
o0_grea = np.array([g0_greater(t)[0, 0] for t in times], dtype=complex)

if not np.allclose(on[0, :, 0], o0_less, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0 less'

if not np.allclose(on[0, :, 1], o0_grea, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0 grea'

if not np.allclose(on_all[0, :, 0], o0_less, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0_all less'

if not np.allclose(on_all[0, :, 1], o0_grea, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0_all grea'

# order 1
rtol = 0.001
atol = 0.01
with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    if not np.array_equal(times, ref['less']['times']):
        raise RuntimeError, 'FAILED: times are different'

    if not np.allclose(on[1, :, 0], ref['less']['o1'], rtol=rtol, atol=atol):
        print 'pn', S.pn
        raise RuntimeError, 'FAILED o1 less'

    if not np.allclose(on[1, :, 1], ref['grea']['o1'], rtol=rtol, atol=atol):
        print 'pn', S.pn
        raise RuntimeError, 'FAILED o1 grea'

    if not np.allclose(on_all[1, :, 0], ref['less']['o1'], rtol=rtol, atol=atol):
        print 'pn', S.pn_all
        raise RuntimeError, 'FAILED o1_all less'

    if not np.allclose(on_all[1, :, 1], ref['grea']['o1'], rtol=rtol, atol=atol):
        print 'pn', S.pn_all
        raise RuntimeError, 'FAILED o1_all grea'

print 'SUCCESS !'
