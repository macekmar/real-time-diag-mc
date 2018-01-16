from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import os

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
p["creation_ops"] = [(0, 0.0, 1)]
p["annihilation_ops"] = []
p["extern_alphas"] = [0.]
p["interaction_start"] = tmax
p["measure_state"] = 0
p["measure_keldysh_indices"] = [0]
p["measure_times"] = [0]
p["U"] = 2.5 # U_qmc
p["max_perturbation_order"] = 7
p["min_perturbation_order"] = 0
p["alpha"] = 0.5
p["length_cycle"] = 10
p["w_ins_rem"] = 1
p["w_dbl"] = 0
p["w_shift"] = 0
p["method"] = 5
p["singular_thresholds"] = [3.0, 3.3]

S = SolverCore(**p)
S.set_g0(g0_lesser, g0_greater)

c0, _ = S.order_zero

S.run(1000, False) # warmup
S.run(5000, True)

S.compute_sn_from_kernels

on = perturbation_series(c0, S.pn, S.sn, p["U"])
on = -1.j * np.squeeze(on)
print on
print

if mpi.is_master_node():
    with HDFArchive('out_files/' + os.path.basename(__file__)[:-3] + '.out.h5', 'w') as ar:
        ar['on'] = on
        ar['times'] = p["measure_times"]

if on.shape != (8,):
    raise RuntimeError, 'FAILED: on shape is ' + str(on.shape) + ' but should be (8,)'

nb_measures = S.nb_measures
if nb_measures != 5000:
    raise RuntimeError, 'FAILED: Solver reported having completed {0} measures instead of 5000'.format(nb_measures)

with HDFArchive('ref_data/density.ref.h5', 'r') as ar:
    on_ref = perturbation_series(abs(g0_lesser(0.)[0, 0]), ar['pn_values'], ar['sn_values'], 2.5)
    print on_ref
    print

print np.abs(on - on_ref)
print

# import matplotlib.pyplot as plt
# plt.plot(on.real, 'b-o')
# plt.plot(on_ref.real, 'b--^')
# plt.plot(on.imag, 'r-o')
# plt.plot(on_ref.imag, 'r--^')
# plt.show()

if not np.allclose(on, on_ref, atol=1e-3):
    raise RuntimeError, 'FAILED density'

print 'SUCCESS !'

