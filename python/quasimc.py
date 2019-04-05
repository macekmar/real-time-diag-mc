from datetime import datetime, timedelta
from time import clock

import numpy as np

import model
from quasimc_utility import distribute_u, distribute_u_complex

from ctint_keldysh import SolverCore
from mpi4py import MPI
from solver import (PARAMS_CPP_KEYS, PARAMS_PYTHON_KEYS,
                    _add_params_to_results, _collect_results, add_cn_to_results,
                    _save_in_file, extract_and_check_params, reduce_binning)
from fourier_transform import fourier_transform

###############################################################################

### Extend params keys for the quasi method
PARAMS_PYTHON_KEYS['N'] = None
PARAMS_PYTHON_KEYS['order'] = None # int or a list of ints
PARAMS_PYTHON_KEYS['model'] = None
PARAMS_PYTHON_KEYS['num_gen_mode'] = 'complex'
PARAMS_PYTHON_KEYS['frequency_range'] = False # False to store times, tuple (freq_min, freq_max, nb_freq) to store frequencies

### Removed not needed
del(PARAMS_PYTHON_KEYS['nb_warmup_cycles'])

def quasi_solver(solver, **params):
    world = MPI.COMM_WORLD

    start_time = datetime.now()

    ### Process parameters
    params_py = extract_and_check_params(params, PARAMS_PYTHON_KEYS)
    params_cpp = extract_and_check_params(params, PARAMS_CPP_KEYS)
    do_fourier = False if (params_py['frequency_range'] is False) else True

    if params_cpp['method'] == 0:
        raise Exception("Method 0 is not implemented yet.")
    t_min = -params_cpp['interaction_start']

    if not((params_py['num_gen_mode'] is 'simple') or (params_py['num_gen_mode'] is 'complex')):
        raise Exception("Number generator mode should be 'simple' or 'complex'.")

    ### result structure
    results_to_save = {}
    metadata = {}
    metadata['duration'] = {}
    metadata['nb_measures'] = {}
    metadata['N'] = []
    # Remove 'model', which cannot be saved, g0_lesser is taken care of in _add_params_to_results
    model_funs = params_py['model'][:]
    del(params_py['model'])
    _add_params_to_results(results_to_save, dict(params_cpp, **params_py))

    ### Calculate inverse CDFs
    integral, inv_cdf = model.calculate_inv_cdfs(model_funs, t_min=t_min)

    N_vec = [0] + params_py['N'][:]
    N_max = N_vec[-1]
    orders = params_py["order"]
    if isinstance(orders, int):
        orders = [orders]

    ### Calculation
    for io, order in enumerate(orders):

        if world.rank == 0:
            print
            print "=================================================================="
            print "Order: {0}".format(order)
            print

        results = {}
        results['results_part'] = {}


        ### Generate quasi random numbers
        points = [None for i in range(len(N_vec)-1)]
        N_true = [0 for i in range(len(N_vec)-1)]
        N_to_calc = [0 for i in range(len(N_vec)-1)]
        if world.rank == 0:
            print 'Generating quasi random numbers'
            if params_py['num_gen_mode'] is 'simple':
                u = model.generate_u(inv_cdf, t_min, N_max, order)
                N_true, N_to_calc, points = distribute_u(u, order, t_min, N_vec, world.size)
            if params_py['num_gen_mode'] is 'complex':
                # # This generates roughly enough points
                u = model.generate_u_complex(inv_cdf, t_min, N_max, order)
                N_true, N_to_calc, points = distribute_u_complex(u, order, t_min, N_vec, world.size)

        N_true = world.bcast(N_true, root=0)
        N_to_calc = world.bcast(N_to_calc, root=0)

        ### Calculate
        N_total = 0
        N_calc = 0
        for iN in range(len(N_vec) - 1):
            if world.rank == 0:
                print "Calculating points from {0} to {1}.".format(N_vec[iN], N_vec[iN+1])
                print "Number of generated points: {0}".format(N_true[iN])
                print "Number of points in the domain (+ trailing zeros): {0}".format(np.sum([p.shape[0] for p in points[iN]]))


            u = world.scatter(points[iN], root=0)
            # Points are padded, remove trailing zeros
            inds = np.where(np.all(u != 0.0, axis=1))[0]
            u = u[inds]

            # Calculate

            # TODO: Implement 'save_period'

            _ = model.get(solver, u, True)
            N_total += N_true[iN]
            N_calc += N_to_calc[iN]

            world.barrier()
            solver.collect_qmc_weight(1) # 1 is just a dummy variable

            ### Process results
            if world.rank == 0:

                print 'Run time:', datetime.now() - start_time
                print 'Date time:', datetime.now()

                res_all = {}
                for key in ['bin_times', 'dirac_times', 'kernels', 'nb_kernels', 'kernel_diracs']:
                    res_all[key] = np.array(getattr(solver, key))
                res_all['U'] = np.array(solver.U)[np.newaxis, :]
                res_all['pn'] = np.zeros((1, params_cpp["max_perturbation_order"] - params_cpp["min_perturbation_order"]))
                res_all['pc'] = np.zeros((1, params_cpp["max_perturbation_order"] - params_cpp["min_perturbation_order"]))
                res_all['pn'][0][order-1] = N_total
                res_all['pc'][0][order-1] = N_calc

                ### Fourier transform if asked
                if do_fourier:
                    w_window = params_py['frequency_range'][0:2]
                    nb_w = params_py['frequency_range'][2]
                    w, kernels_w = fourier_transform(res_all['bin_times'], res_all['kernels'],
                                                     w_window, nb_w, axis=1)
                    res_all['omega'] = w
                    res_all['kernels'] = kernels_w

                    del res_all['bin_times']
                    del res_all['nb_kernels']

                # res_all
                results['results_all'] = res_all # Lower orders are still contained in res_all

                # res_part
                res_part = dict(res_all)
                # bin reduction
                if ('kernels' in res_part) and (not do_fourier):
                    nb_bins_sum = params_py['nb_bins_sum']
                    res_part['bin_times'] = reduce_binning(res_part['bin_times'], nb_bins_sum) / float(nb_bins_sum)
                    res_part['kernels'] = reduce_binning(res_part['kernels'], nb_bins_sum, axis=1) / float(nb_bins_sum)
                    res_part['nb_kernels'] = reduce_binning(res_part['nb_kernels'], nb_bins_sum, axis=1) # no normalization !

                for key in results['results_all']:
                    if key not in ["U", "bin_times", "omega"]:
                        if iN == 0:
                            if key == "pn" or key == 'pc':
                                results['results_part'][key] = np.zeros([res_part[key].shape[1]] + [len(N_vec)-1], dtype=res_part[key].dtype)
                            else:
                                results['results_part'][key] = np.zeros(list(res_part[key].shape) + [len(N_vec)-1], dtype=res_part[key].dtype)
                                # results['results_part'][key] = res_part[key][..., np.newaxis]
                            results['results_part'][key][..., iN] = res_part[key]
                        else:
                            # results['results_part'][key] = np.concatenate((results['results_part'][key], res_part[key][..., np.newaxis]), axis=-1)
                            results['results_part'][key][..., iN] = res_part[key]
                    else:
                        results['results_part'][key] = results['results_all'][key]

                # ### metadata that are different in each subrun
                # metadata['duration'][order][N_total] = solver.qmc_duration
                # metadata['nb_measures'][order][N_total] = solver.nb_measures
                # metadata['nb_proc'] = MPI.COMM_WORLD.size
                # metadata['N'].append(N_total)
                # metadata['N_demanded'] = N_vec[:iN+1]
                results['metadata'] = metadata

                if io == 0 and iN == 0:
                    for key in results:
                        results_to_save[key] = results[key]
                    cn = [np.prod(integral[:i+1]) for i in range(len(integral))]
                    cn = np.insert(cn, 0, 1)
                    # results_to_save["results_all"]["cn"] = np.ones(params_cpp["max_perturbation_order"]+1)
                    # results_to_save["results_part"]["cn"] = np.ones((params_cpp["max_perturbation_order"]+1, iN+1))
                    results_to_save["results_all"]["cn"] = cn
                    #results_to_save["results_all"]["cn"][1:] /= results_to_save["results_all"]["pn"][0]
                    results_to_save["results_part"]["cn"] = cn[:, np.newaxis] #np.tile(cn, (len(N_vec), 1))
                else:
                    results_to_save = quasi_merge_results(order, results_to_save, results)

                # Modify pn
                # res = dict(results_to_save)
                # res["results_all"]["pn"] = res["results_all"]["pn"][np.newaxis,:]
                _save_in_file(results_to_save, params_py['filename'], params_py['run_name'])

    return results

def quasi_merge_results(order, results_to_save, results):
    for res in ["results_all", "results_part"]:
        for key in ["kernels", 'nb_kernels', 'kernel_diracs']:
            try:
                results_to_save[res][key][order-1] = results[res][key][order-1]
            except KeyError: # nb_kernels is absent if kernels have been Fourier transformed
                pass

    for k in ['pn', 'pc']:
        res, key = 'results_all', k
        results_to_save[res][key][0][order-1] = results[res][key][0][order-1]
        res, key = 'results_part', k
        results_to_save[res][key][order-1] = results[res][key][order-1]

    return results_to_save
