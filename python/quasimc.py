import sys
from datetime import datetime

import numpy as np
from mpi4py import MPI

from fourier_transform import fourier_transform
from quasimc_results import *
from quasimc_utility import calculate_inv_cdfs, process_parameters
from solver import PARAMS_CPP_KEYS, PARAMS_PYTHON_KEYS, _add_params_to_results

###############################################################################

### Extend params keys for the quasi method
PARAMS_PYTHON_KEYS['N'] = None
PARAMS_PYTHON_KEYS['order'] = None # int or a list of ints
PARAMS_PYTHON_KEYS['model'] = None
PARAMS_PYTHON_KEYS['frequency_range'] = False # False to store times, tuple (freq_min, freq_max, nb_freq) to store frequencies

PARAMS_PYTHON_KEYS['num_gen'] = None
PARAMS_PYTHON_KEYS['random_shift'] = False
PARAMS_PYTHON_KEYS['random_seed'] = False
PARAMS_PYTHON_KEYS['filemode'] = 'w' # Filemode for opening the hdf files, 'w' or 'a'
PARAMS_PYTHON_KEYS['interpolation_points'] = False
PARAMS_PYTHON_KEYS['keep_cubic_domain'] = True
PARAMS_PYTHON_KEYS['baker_transform'] = False

### Removed not needed
del PARAMS_PYTHON_KEYS['nb_warmup_cycles']
del PARAMS_PYTHON_KEYS['staircase']
del PARAMS_PYTHON_KEYS['nb_cycles']
del PARAMS_PYTHON_KEYS['g0_lesser']
del PARAMS_PYTHON_KEYS['g0_greater']

def quasi_solver(solver, **params):
    world = MPI.COMM_WORLD

    start_time = datetime.now()

    t_start, t_g0_max, model_funs, intp_pts, gen_class, nb_bins_sum, random_shift, seed, \
    save_period, overwrite, params_py, params_cpp = \
                process_parameters(params, PARAMS_PYTHON_KEYS, PARAMS_CPP_KEYS)
    if params_py['keep_cubic_domain']:
        t_lim = t_start
    else:
        t_lim = t_g0_max

    orders = params_py["order"]
    if isinstance(orders, int):
        orders = [orders]

    if not isinstance(params_py["N"][0], list):
        N_vec_orders = [params_py['N'][:] for o in orders]
    else:
        assert len(params_py['N']) == len(orders)
        for i in range(len(params_py['N'])):
            assert len(params_py['N'][i]) == len(params_py['N'][0])
        N_vec_orders = params_py['N'][:]

    for i in range(len(N_vec_orders)):
        N = N_vec_orders[i][:]
        N = [0] + N
        N = list(set(N)) # Remove duplicates
        N.sort()
        N_vec_orders[i] = N

    ### Calculate inverse CDFs and the integral of the model
    # Integrals are not needed for normalization, because the model is already
    # normalized.
    integral, inv_cdf = calculate_inv_cdfs(model_funs, intp_pts)
    model_ints = [np.prod(integral[:i+1]) for i in range(len(integral))]
    model_ints = np.insert(model_ints, 0, 1)
    # Set model
    intervals = [inv_cdf[i].x.tolist() for i in range(max(params_py['order']))]
    coeff = [inv_cdf[i].c.T.tolist() for i in range(max(params_py['order']))]
    solver.set_model(intervals, coeff)

    ### Prepare results
    results_to_save = create_empty_results(orders, N_vec_orders[0], params_py, params_cpp)
    _add_params_to_results(results_to_save, dict(params_cpp, **params_py))
    metadata = {}
    metadata['total_duration'] = 0.0
    metadata['order_duration'] = np.empty(len(orders), dtype=np.float)
    metadata['nb_proc'] = MPI.COMM_WORLD.size
    metadata['orders'] = orders
    metadata['model_integrals'] = model_ints
    metadata['random_shift'] = random_shift
    metadata['random_num_generator'] = str(gen_class)
    metadata['random_seed'] = seed
    metadata['sampling_coeff'] = coeff
    metadata['sampling_intervals'] = intervals
    results_to_save['metadata'] = metadata
    if world.rank == 0:
        params_py['run_name'] = save_empty_results(results_to_save, params_py['filename'], params_py['run_name'], overwrite=overwrite, filemode=params_py['filemode'])
        print "Saving into run_name: %s" % params_py['run_name']

    ### Calculation
    for io, order in enumerate(orders):
        order_start_time = datetime.now()

        if world.rank == 0:
            print "\n=================================================================="
            print "Order: {0}\n".format(order)

        
        generator = gen_class(dim=order, seed=seed)

        ### Calculate
        N_generated = 0
        N_calculated = 0
        N_vec = N_vec_orders[io]
        for iN in range(len(N_vec) - 1):
            if world.rank == 0:
                print "\nCalculating points from {0} to {1}.".format(world.size*N_vec[iN], world.size*N_vec[iN+1])

            for _ in range(N_vec[iN+1] - N_vec[iN]):
                # Generate l whose u lies inside of the domain
                l = []
                while len(l) < world.size:
                    N_generated += 1
                    l_proposed = generator.next()
                    l_proposed = (l_proposed + random_shift[:order]) % 1
                    if params_py['baker_transform'] is True:
                        l_proposed= np.where(l_proposed < 0.5, 2*l_proposed, 2-2*l_proposed)
                    u = solver.l_to_u([float(x) for x in l_proposed])
                    # check if in domain
                    if np.all(np.array(u) > -t_lim): # u are negative
                        l.append(l_proposed)

                # each rank only calculates the rank-th point out of world.size number of them
                solver.evaluate_importance_sampling([float(x) for x in l[world.rank]], True)

                N_calculated += 1
                ### Process and save results
                if N_calculated % save_period[io] == 0 or N_calculated == N_vec[iN+1]:
                    solver.collect_sampling_weights(1) # world.barrier is inside measure.collect_results()
                    if world.rank == 0:
                        if N_calculated != N_vec[iN+1]:
                            ratio_calc = (N_calculated - N_vec[iN])/float(N_vec[iN+1] - N_vec[iN])
                            sys.stdout.write("% 2.0f%% " % (100*ratio_calc))

                        chunk_results = extract_results(solver, params_cpp)
                        chunk_results['N_generated'] = N_generated
                        chunk_results['N_calculated'] = world.size*N_calculated
                        # # Normalization
                        #normalize_results(chunk_results, order, N_generated, params_cpp) # Not needed anymore. Why?! Because from commit b479be on we init_measure only once? I guess so...
                        metadata['total_duration'] = (datetime.now() - start_time).total_seconds()
                        metadata['order_duration'] = (datetime.now() - order_start_time).total_seconds()
                        update_results(chunk_results, metadata, io, order, iN, nb_bins_sum, params_py, params_cpp)

            ### Print some results at the end of each N_vec:
            if world.rank == 0:
                print '\nDate time:', datetime.now()
                print 'Total run time:', datetime.now() - start_time, ' Order run time:', datetime.now() - order_start_time
                print 'Total demanded points: %d*%d  Gen. points: %d Calc. pts: %d' % (world.size, N_vec[iN+1],  N_generated, world.size*N_calculated)

    return solver