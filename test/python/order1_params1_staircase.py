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
p["right_input_points"] = [(0, 0.0, 0)]
p["interaction_start"] = 40.0
p["measure_state"] = 0
p["measure_keldysh_indices"] = [0, 1]
p["measure_times"] = times
p["U"] = [2.5, 2.5] # U_qmc
p["min_perturbation_order"] = 0
p["max_perturbation_order"] = 2
p["alpha"] = 0.0
p["n_warmup_cycles"] = 1000
p["n_cycles"] = 100000
p["length_cycle"] = 50
p["w_dbl"] = 0
p["w_shift"] = 0
p["method"] = 5
p["singular_thresholds"] = [3.0, 3.3]

on, on_error = staircase_solve(g0_lesser, g0_greater, p)


if mpi.is_master_node():
    with HDFArchive('out_files/' + os.path.basename(__file__)[:-3] + '.out.h5', 'w') as ar:
        ar['on'] = on
        ar['on_error'] = on_error
        ar['times'] = times

if on.shape != (3, 101, 2):
    raise RuntimeError, 'FAILED: on shape is ' + str(on.shape) + ' but should be (3, 101, 2)'

if on_error.shape != (3, 101, 2):
    raise RuntimeError, 'FAILED: on_error shape is ' + str(on.shape) + ' but should be (3, 101, 2)'


# order 0
rtol = 0.001
atol = 0.0001
o0_less = np.array([g0_lesser(t)[0, 0] for t in times], dtype=complex)
o0_grea = np.array([g0_greater(t)[0, 0] for t in times], dtype=complex)

if not np.allclose(on[0, :, 0], o0_less, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0 less'

if not np.allclose(on[0, :, 1], o0_grea, rtol=rtol, atol=atol):
    raise RuntimeError, 'FAILED o0 grea'

# order 1
rtol = 0.001
atol = 0.01
with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    if not np.array_equal(times, ref['less']['times']):
        raise RuntimeError, 'FAILED: times are different'

    if not np.allclose(on[1, :, 0], ref['less']['o1'], rtol=rtol, atol=atol):
        raise RuntimeError, 'FAILED o1 less'

    if not np.allclose(on[1, :, 1], ref['grea']['o1'], rtol=rtol, atol=atol):
        raise RuntimeError, 'FAILED o1 grea'

