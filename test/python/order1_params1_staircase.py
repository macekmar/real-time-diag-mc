from pytriqs.utility import mpi
from ctint_keldysh import make_g0_semi_circular, solve
from ctint_keldysh.construct_gf import kernel_GF_from_archive
from pytriqs.archive import HDFArchive
import numpy as np
import os
from matplotlib import pyplot as plt

p = {}
p["beta"] = 200.0
p["Gamma"] = 0.5
p["tmax_gf0"] = 100.0
p["Nt_gf0"] = 25000
p["epsilon_d"] = 0.5
p["muL"] = 0.
p["muR"] = 0.

g0_less_triqs, g0_grea_triqs = make_g0_semi_circular(**p)

times = np.linspace(-40.0, 0.0, 101)
filename = 'out_files/' + os.path.basename(__file__)[:-3] + '.out.h5'
p = {}

p["staircase"] = True
p["nb_warmup_cycles"] = 1000
p["nb_cycles"] = 10000
p["save_period"] = 60
p["filename"] = filename
p["g0_lesser"] = g0_less_triqs
p["g0_greater"] = g0_grea_triqs
p["size_part"] = 1
p["nb_bins_sum"] = 10

p["creation_ops"] = [(0, 0, 0.0, 0)]
p["annihilation_ops"] = [(0, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False
p["interaction_start"] = 40.0
p["alpha"] = 0.0
p["nb_orbitals"] = 1

p["U"] = 2.5 # U_qmc
p["w_ins_rem"] = 1.
p["w_dbl"] = 0
p["w_shift"] = 0
p["max_perturbation_order"] = 2
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 50
p["method"] = 5
p["singular_thresholds"] = [3.5, 3.3]

results = solve(p)

with HDFArchive(filename, 'r') as ar:
    kernel = kernel_GF_from_archive(ar, 'kernels', p["nonfixed_op"])

g0_less = np.vectorize(lambda t: g0_less_triqs(t)[0, 0] if np.abs(t) < 100. else 0.)
g0_grea = np.vectorize(lambda t: g0_grea_triqs(t)[0, 0] if np.abs(t) < 100. else 0.)
GF = kernel.convol_g_left(g0_less, g0_grea, 100.)
GF.apply_gf_sym()
GF.increment_order(g0_less, g0_grea)

plt.plot(GF.times, GF.less.values[0].real, 'b')
plt.plot(GF.times, GF.less.values[0].imag, 'r')
plt.plot(GF.times, g0_less(GF.times).real, 'g')
plt.plot(GF.times, g0_less(GF.times).imag, 'm')
plt.show()

with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    plt.plot(ref['less']['times'], ref['less']['o1'].real, 'g.-')
    plt.plot(ref['less']['times'], ref['less']['o1'].imag, 'm.-')
plt.plot(GF.times, GF.less.values[1].real, 'b')
plt.plot(GF.times, GF.less.values[1].imag, 'r')
plt.xlim(-50, 10)
plt.show()

with HDFArchive('ref_data/order1_params1.ref.h5', 'r') as ref:
    plt.plot(ref['grea']['times'], ref['grea']['o1'].real, 'g.-')
    plt.plot(ref['grea']['times'], ref['grea']['o1'].imag, 'm.-')
plt.plot(GF.times, GF.grea.values[1].real, 'b')
plt.plot(GF.times, GF.grea.values[1].imag, 'r')
plt.xlim(-50, 10)
plt.show()

with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
    plt.plot(ref['less']['times'], ref['less']['o2'].real, 'go')
    plt.plot(ref['less']['times'], ref['less']['o2'].imag, 'mo')
plt.plot(GF.times, GF.less.values[2].real, 'b')
plt.plot(GF.times, GF.less.values[2].imag, 'r')
plt.xlim(-50, 10)
plt.show()

with HDFArchive('ref_data/order2_params1.ref.h5', 'r') as ref:
    plt.plot(ref['grea']['times'], ref['grea']['o2'].real, 'go')
    plt.plot(ref['grea']['times'], ref['grea']['o2'].imag, 'mo')
plt.plot(GF.times, GF.grea.values[2].real, 'b')
plt.plot(GF.times, GF.grea.values[2].imag, 'r')
plt.xlim(-50, 10)
plt.show()

exit()

if on.shape != (3, 101, 2):
    raise RuntimeError, 'FAILED: on shape is ' + str(on.shape) + ' but should be (3, 101, 2)'

if on_error.shape != (3, 101, 2):
    raise RuntimeError, 'FAILED: on_error shape is ' + str(on.shape) + ' but should be (3, 101, 2)'

if mpi.world.rank == 0:
    # order 0
    rtol = 0.001
    atol = 0.0001
    o0_less = np.array([g0_less_triqs(t)[0, 0] for t in times], dtype=complex)
    o0_grea = np.array([g0_grea_triqs(t)[0, 0] for t in times], dtype=complex)

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

    # test archive contents
    with HDFArchive(filename, 'r') as ar:
        nb_measures = ar['metadata']['nb_measures']
        if nb_measures[0] != 100000 or nb_measures[1] != 100000:
            print nb_measures
            raise RuntimeError, 'FAILED: Solver reported number of measures != 100000'


    print 'SUCCESS !'
