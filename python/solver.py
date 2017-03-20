# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series, staircase_perturbation_series_cum
import numpy as np
import itertools
from pytriqs.archive import HDFArchive
from datetime import datetime

def variance_error(on, comm=MPI.COMM_WORLD):
    if comm.rank != 0:
        comm.gather(on)
        on_error = comm.bcast(None)
    else:
        on_all = np.array(comm.gather(on), dtype=complex)

        on_error = np.sqrt(np.var(on_all, axis=0) / comm.size)
        comm.bcast(on_error)

    return on_error

def gather_histogram(solver, comm=MPI.COMM_WORLD):
    list_config = solver.config_list
    weight_config = solver.config_weight
    list_config = map(lambda x, y : [x] + list(y), weight_config, list_config)

    # Sort out the configurations
    list_config.sort(key=len)
    list_histograms = []
    for k, g in itertools.groupby(list_config, len):
        list_histograms.append(np.array(list(g)).T)

    list_histograms_all = []

    for histogram in list_histograms:
        if comm.rank != 0:
            comm.gather(histogram)
        else:
            list_histograms_all.append(np.concatenate(comm.gather(histogram), axis=1))

    if comm.rank != 0:
        list_histograms_all = comm.bcast(None)
    else:
        comm.bcast(list_histograms_all)

    return list_histograms_all

def wright(duration, nb_measures, parameters, on, on_error, filename, kind_list, nb_proc):
    # before using save, check kind_list has the good shape
    with HDFArchive(filename, 'w') as ar:
        ar['nb_proc'] = nb_proc
        ar['interaction_start'] = parameters['interaction_start']
        ar['run_time'] = duration
        ar['nb_measures'] = nb_measures

        for i, kind in enumerate(kind_list):
            ar.create_group(kind)
            group = ar[kind]
            group['times'] = parameters['measure_times']
            group['on'] = np.squeeze(on[:, :, i])
            group['on_error'] = np.squeeze(on_error[:, :, i])

    print 'Saved in', filename

def wright_add(data, data_name, filename):
    with HDFArchive(filename, 'a') as ar:
        ar[data_name] = data


######################### Front End #######################

def single_solve(g0_lesser, g0_greater, parameters, filename=None, kind_list=None, save_period=-1, histogram=False):
    save = filename is not None

    if save and len(parameters['measure_keldysh_indices']) != len(kind_list) :
        raise ValueError, 'kind_list has wrong shape'

    world = MPI.COMM_WORLD

    S = SolverCore(**parameters)
    S.set_g0(g0_lesser, g0_greater)

    c0, _ = S.order_zero

    status = 1
    while status == 1:
        status = S.run(save_period) # status=0 if finished, 1 if timed up, 2 if aborted

        on_result = perturbation_series(c0, S.pn_all, S.sn_all, parameters['U'])

        # Calculate error
        # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
        on = perturbation_series(c0, S.pn, S.sn, parameters['U'])
        on_error = variance_error(on, world)

        if save and world.rank == 0:
            wright(S.solve_duration, S.nb_measures_all, parameters, on, on_error, filename, kind_list, world.size)
            wright_add(S.kernels_all, 'kernels', filename)

        output = (np.squeeze(on), np.squeeze(on_error))
        if world.rank == 0:
            print "Number of measures (all procs):", S.nb_measures_all
            print

        if histogram:
            list_hist = gather_histogram(S, world)
            output += (list_hist,)

            if world.rank == 0:
                with HDFArchive(filename, 'a') as ar:
                    ar['histogram'] = list_hist

    return output


def staircase_solve(g0_lesser, g0_greater, _parameters, filename=None, kind_list=None, save_period=-1, only_even=False):
    save = filename is not None

    if save and len(_parameters['measure_keldysh_indices']) != len(kind_list) :
        raise ValueError, 'kind_list has wrong shape'

    world = MPI.COMM_WORLD

    if world.rank == 0:
        print "\n----------- Save staircase solver -----------"

    parameters = _parameters.copy()
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    solve_duration = 0
    nb_measures = 0

    if not only_even:
        orders = list(range(1, max_order + 1))
    else:
        orders = list(range(2, max_order + 1, 2))

    pn_all = (len(orders) + 1) * [[]]
    pn     = (len(orders) + 1) * [[]]
    sn_all = (len(orders) + 1) * [[]]
    sn     = (len(orders) + 1) * [[]]
    kernels = (len(orders) + 1) * [[]]
    nb_kernels = (len(orders) + 1) * [[]]

    # order zero
    S = SolverCore(**parameters)
    S.set_g0(g0_lesser, g0_greater)
    c0, s0 = S.order_zero

    pn_all[0] = [1]
    pn[0] = [1]
    sn_all[0] = s0[np.newaxis, :]
    sn[0] = s0[np.newaxis, :]

    for i, k in enumerate(orders):
        i += 1

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k

        S = SolverCore(**parameters)
        S.set_g0(g0_lesser, g0_greater)

        status = 1
        while status == 1:
            status = S.run(save_period)

            pn_all[i] = S.pn_all
            pn[i] = S.pn
            sn_all[i] = S.sn_all
            sn[i] = S.sn
            # TODO: kernels are stored like a mess, do something
            kernels[i] = S.kernels_all[k, ...]
            nb_kernels[i] = S.nb_values[k, ...]

            if world.rank == 0:
                for i_print in range(i+1):
                    print pn_all[i_print]
                print

            # calculates ideal U value
            pn_nonzero = S.pn_all
            nonzero_ind = np.nonzero(pn_nonzero)[0]
            pn_nonzero = pn_nonzero[nonzero_ind].astype(np.float32)
            if len(pn_nonzero) >= 2:
                power = float(nonzero_ind[-1] - nonzero_ind[-2])
                U_proposed = parameters['U'] * pow(pn_nonzero[-2] / pn_nonzero[-1], 1. / power)
            else:
                U_proposed = None

            # calculates results
            on_result = staircase_perturbation_series(c0, pn_all[:i+1], sn_all[:i+1], parameters['U'])

            # Calculate error
            # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
            on = staircase_perturbation_series(c0, pn[:i+1], sn[:i+1], parameters['U'])
            on_error = variance_error(on, world)

            if world.rank == 0:
                print datetime.now(), ": Order", k
                print "Duration (this orders):", S.solve_duration
                print "Duration (all orders):", solve_duration + S.solve_duration
                print "Number of measures (all procs, this order):", S.nb_measures_all
                print "Number of measures (all procs, all orders):", nb_measures + S.nb_measures_all
                print "U proposed:", U_proposed, "(current is", parameters['U'], ")"
                print
                if save:
                    wright(solve_duration + S.solve_duration, nb_measures + S.nb_measures_all, parameters, on_result[:k+1], on_error[:k+1], filename, kind_list, world.size)
                    wright_add(kernels, 'kernels', filename)
                    wright_add(nb_kernels, 'nb_kernels', filename)

        solve_duration += S.solve_duration
        nb_measures += S.nb_measures_all
        if status == 2: break # Received signal, terminate

    return np.squeeze(on_result), np.squeeze(on_error)

