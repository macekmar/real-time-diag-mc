from pytriqs.utility import mpi
from ctint_keldysh import make_g0_lattice_1d, solve
from ctint_keldysh.construct_gf import kernel_GF_from_archive
from pytriqs.archive import HDFArchive
import numpy as np
import os
from matplotlib import pyplot as plt
from copy import deepcopy

from toolbox import cpx_plot_error

nb_sites = 11
site_of_int = 5
cutoff_g0 = 200.
p = {}
p['beta'] = 1.
p['mu'] = 0.5
p['epsilon'] = 0.
p['hop'] = 1.
p['tmax_gf0'] = cutoff_g0
p['Nt_gf0'] = 2000
p['nb_sites'] = nb_sites
p['Nb_k_pts'] = 300

if mpi.world.rank == 0:
    print 'Generating g0 ...'
g0_less, g0_grea = make_g0_lattice_1d(**p)
if mpi.world.rank == 0:
    print 'done.'

filename = 'out_files/' + os.path.basename(__file__)[:-3] + '.out.h5'
p = {}

p["staircase"] = True
p["nb_warmup_cycles"] = 1000
p["nb_cycles"] = 10000#0
p["save_period"] = 3*60
p["filename"] = filename
p["run_name"] = 'run_1'
p["g0_lesser"] = g0_less
p["g0_greater"] = g0_grea
p["size_part"] = 10
p["nb_bins_sum"] = 10

p["creation_ops"] = [(site_of_int, 0, 0.0, 0)]
p["annihilation_ops"] = [(site_of_int, 0, 0.0, 0)]
p["extern_alphas"] = [0.]
p["nonfixed_op"] = False # annihilation
p["interaction_start"] = 100.
p["alpha"] = 0.0
p["nb_orbitals"] = nb_sites
p["potential"] = (nb_sites * [1.], list(range(nb_sites)), list(range(nb_sites)))

p["U"] = 0.16 # U_qmc
p["w_ins_rem"] = 0.5
p["w_dbl"] = 0.
p["w_shift"] = 1.
p["max_perturbation_order"] = 3
p["min_perturbation_order"] = 0
p["forbid_parity_order"] = -1
p["length_cycle"] = 50
p["method"] = 1
p["singular_thresholds"] = [3.5, 3.3]

p_copy = deepcopy(p) # keep parameters safe

# solve(**p)

if mpi.world.rank == 0:
    g0_less_v = np.vectorize(lambda t: g0_less(t)[0, 0] if np.abs(t) < cutoff_g0 else 0., otypes=[complex])
    g0_grea_v = np.vectorize(lambda t: g0_grea(t)[0, 0] if np.abs(t) < cutoff_g0 else 0., otypes=[complex])

    with HDFArchive(filename, 'r') as ar:

        kernel = kernel_GF_from_archive(ar['run_1']['results_all'], p_copy["nonfixed_op"], site_of_int)
        kernel_part = kernel_GF_from_archive(ar['run_1']['results_part'], p_copy["nonfixed_op"], site_of_int)

    GF = kernel.convol_g_left(g0_less_v, g0_grea_v, cutoff_g0)
    GF.apply_gf_sym()
    GF.increment_order(g0_less_v, g0_grea_v)

    GF_part = kernel_part.convol_g_left(g0_less_v, g0_grea_v, cutoff_g0)
    GF_part.apply_gf_sym()
    GF_part.increment_order(g0_less_v, g0_grea_v)
    error = lambda a : np.sqrt(p_copy['nb_bins_sum'] * np.var(a, axis=-1) / float(a.shape[-1]))

    ### lesser
    fig, ax = plt.subplots(p_copy['max_perturbation_order'] + 1, 1)
    for k in range(len(ax)):
        cpx_plot_error(ax[k], GF.less.times, GF.less.values[k], GF_part.less.times,
                (error(GF_part.less.values[k].real), error(GF_part.less.values[k].imag)),
                color=('b', 'r'))
    plt.show()

    GFw = GF.fourier_transform(max_puls=5., min_data_pts=1000)
    GFw_part = GF_part.fourier_transform(max_puls=5., min_data_pts=500)

    fig, ax = plt.subplots(p_copy['max_perturbation_order'] + 1, 1)
    for k in range(len(ax)):
        cpx_plot_error(ax[k], GFw.less.puls, GFw.less.values[k], GFw_part.less.puls,
                (error(GFw_part.less.values[k].real), error(GFw_part.less.values[k].imag)),
                color=('b', 'r'))
    plt.show()
    ### retarded
    RGF = GF.retarded()
    RGF_part = GF_part.retarded()

    fig, ax = plt.subplots(p_copy['max_perturbation_order'] + 1, 1)
    for k in range(len(ax)):
        cpx_plot_error(ax[k], RGF.times, RGF.values[k], RGF_part.times,
                (error(RGF_part.values[k].real), error(RGF_part.values[k].imag)),
                color=('b', 'r'))
    plt.show()

    RGF = RGF.fourier_transform(max_puls=5., min_data_pts=1000)
    RGF_part = RGF_part.fourier_transform(max_puls=5., min_data_pts=500)

    fig, ax = plt.subplots(p_copy['max_perturbation_order'] + 1, 1)
    for k in range(len(ax)):
        cpx_plot_error(ax[k], RGF.puls, RGF.values[k], RGF_part.puls,
                (error(RGF_part.values[k].real), error(RGF_part.values[k].imag)),
                color=('b', 'r'))
    plt.show()
