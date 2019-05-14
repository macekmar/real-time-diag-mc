import sys
from datetime import datetime, timedelta
from time import clock

import numpy as np

import model
from fourier_transform import fourier_transform
from mpi4py import MPI
from quasimc_results import *
from quasimc_utility import process_parameters
from solver import PARAMS_CPP_KEYS, PARAMS_PYTHON_KEYS, _add_params_to_results

###############################################################################

### Extend params keys for the quasi method
PARAMS_PYTHON_KEYS['N'] = None
PARAMS_PYTHON_KEYS['order'] = None # int or a list of ints
PARAMS_PYTHON_KEYS['model'] = None
PARAMS_PYTHON_KEYS['frequency_range'] = False # False to store times, tuple (freq_min, freq_max, nb_freq) to store frequencies

PARAMS_PYTHON_KEYS['num_gen_mode'] = 'complex'
PARAMS_PYTHON_KEYS['num_gen'] = None
PARAMS_PYTHON_KEYS['random_shift'] = 0.0
PARAMS_PYTHON_KEYS['filemode'] = 'w' # Filemode for opening the hdf files, 'w' or 'a'

### Removed not needed
del(PARAMS_PYTHON_KEYS['nb_warmup_cycles'])

def quasi_solver(solver, **params):
    world = MPI.COMM_WORLD

    start_time = datetime.now()

    t_min, model_funs, gen_class, nb_bins_sum, params_py, params_cpp = \
                process_parameters(params, PARAMS_PYTHON_KEYS, PARAMS_CPP_KEYS)
    
    N_vec = [0] + params_py['N'][:]
    orders = params_py["order"]
    if isinstance(orders, int):
        orders = [orders]

    ### Calculate inverse CDFs and the integral of the model
    integral, inv_cdf = model.calculate_inv_cdfs(model_funs, t_min=t_min)
    model_ints = [np.prod(integral[:i+1]) for i in range(len(integral))]
    model_ints = np.insert(model_ints, 0, 1)

    ### Prepare results
    results_to_save = create_empty_results(orders, N_vec, params_py, params_cpp)
    _add_params_to_results(results_to_save, dict(params_cpp, **params_py))
    metadata = {}
    metadata['total_duration'] = 0.0
    metadata['order_duration'] = np.empty(len(orders), dtype=np.float)
    metadata['nb_proc'] = MPI.COMM_WORLD.size
    metadata['orders'] = orders
    metadata['model_integrals'] = model_ints
    results_to_save['metadata'] = metadata
    # TODO: generator, seed, random shift
    if world.rank == 0:
        params_py['run_name'] = save_empty_results(results_to_save, params_py['filename'], params_py['run_name'], params_py['filemode'])

    ### Calculation
    last_save = datetime.now()
    for io, order in enumerate(orders):
        order_start_time = datetime.now()

        if world.rank == 0:
            print "\n=================================================================="
            print "Order: {0}\n".format(order)

        generator = gen_class(dim=order, seed=1<<30)  # TODO seed

        ### Calculate
        N_generated = 0
        N_calculated = 0
        for iN in range(len(N_vec) - 1):
            if world.rank == 0:
                print "\nCalculating points from {0} to {1}.".format(N_vec[iN], N_vec[iN+1])
       
            itr = 0
            for l in generator:
                if N_calculated == N_vec[iN+1]:
                    break
                # v += shift # TODO
                v = model.l_to_v(inv_cdf, l[np.newaxis,:])[0]
                u = np.array(v, dtype=np.float)
                N_generated += 1
                # check if domain
                if np.all(u > t_min):
                    itr += 1
                    if itr % world.size != world.rank:
                        continue           
                    # calculate
                    solver.evaluate_importance_sampling([float(x) for x in l], True)

                    N_calculated += 1
                ### Process and save results
                # save above save_period or at the end of each N_vec
                time_from_save = datetime.now() - last_save
                if time_from_save.total_seconds() > params_py['save_period'] or \
                                                    N_calculated == N_vec[iN+1]:
                    world.barrier()
                    solver.collect_sampling_weights(1)
                    if world.rank == 0:
                        if N_calculated != N_vec[iN+1]:
                            ratio_calc = (N_calculated - N_vec[iN])/float(N_vec[iN+1] - N_vec[iN])
                            sys.stdout.write("% 2.0f%% " % (100*ratio_calc))
                        
                        chunk_results = extract_results(solver)
                        chunk_results['N_generated'] = N_generated
                        chunk_results['N_calculated'] = world.size*N_calculated
                        # Normalization
                        chunk_results['kernels'][order-1] *= model_ints[order]/float(N_generated)

                        metadata['total_duration'] = (datetime.now() - start_time).total_seconds()
                        metadata['order_duration'] = (datetime.now() - order_start_time).total_seconds()

                        update_results(chunk_results, metadata, io, order, iN, nb_bins_sum, params_py)
                        
                    world.barrier()
                    last_save = datetime.now()

            ### Print some results at the end of each N_vec:
            if world.rank == 0:
                print '\nDate time:', datetime.now()
                print 'Total run time:', datetime.now() - start_time, ' Order run time:', datetime.now() - order_start_time
                print 'Demanded points: %d  Gen. points: %d Calc. pts: %d' % (N_vec[iN+1],  N_generated, world.size*N_calculated)

