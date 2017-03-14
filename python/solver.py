# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series
import numpy as np
import itertools
from pytriqs.archive import HDFArchive

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

def save(duration, nb_measures, parameters, on, on_error, filename, kind_list, nb_proc):
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

def save_add(data, data_name, filename):
    with HDFArchive(filename, 'a') as ar:
        ar[data_name] = data


######################### Front End #######################

def single_solve(g0_lesser, g0_greater, parameters, max_time=-1, histogram=False):

    world = MPI.COMM_WORLD

    S = SolverCore(**parameters)
    S.set_g0(g0_lesser, g0_greater)

    c0, _ = S.order_zero
    S.run(max_time)

    on_result = perturbation_series(c0, S.pn_all, S.sn_all, parameters['U'])

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = perturbation_series(c0, S.pn, S.sn, parameters['U'])
    on_error = variance_error(on, world)
    output = (np.squeeze(on), np.squeeze(on_error))
    print "Number of measures (all procs):", S.nb_measures_all
    print

    if histogram:
        list_hist = gather_histogram(S, world)
        output += (list_hist,)

    return output

def save_single_solve(g0_lesser, g0_greater, parameters, filename, kind_list, save_period=-1, histogram=False):

    if len(parameters['measure_keldysh_indices']) != len(kind_list) :
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

        if world.rank == 0:
            save(S.solve_duration, S.nb_measures_all, parameters, on, on_error, filename, kind_list, world.size)
            save_add(S.kernels_all, 'kernels', filename)

        output = (np.squeeze(on), np.squeeze(on_error))
        print "Number of measures (all procs):", S.nb_measures_all
        print

        if histogram:
            list_hist = gather_histogram(S, world)
            output += (list_hist,)

            if world.rank == 0:
                with HDFArchive(filename, 'a') as ar:
                    ar['histogram'] = list_hist

    return output


def staircase_solve(g0_lesser, g0_greater, _parameters, max_time=-1):

    world = MPI.COMM_WORLD

    if world.rank == 0:
        print "\n----------- Staircase solver -----------"

    parameters = _parameters.copy()
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    solve_duration = 0

    for k in range(1, max_order + 1):

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k

        S = SolverCore(**parameters)
        S.set_g0(g0_lesser, g0_greater)

        if k == 1:
            # order zero and initializing arrays
            c0, s0 = S.order_zero

            # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
            pn_all = np.empty((max_order+1, max_order+1), dtype=int) * np.nan
            pn = pn_all.copy()
            sn_all = np.empty(pn_all.shape + s0.shape, dtype=complex) * np.nan
            sn = sn_all.copy()

            pn_all[0, 0] = 1 # needed to avoid NaNs, any non zero value works
            pn[0, 0] = 1
            sn_all[0, 0, ...] = s0
            sn[0, 0, ...] = s0

        status = S.run(max_time - solve_duration)
        solve_duration += S.solve_duration

        pn_all[k, :k+1] = S.pn_all
        pn[k, :k+1] = S.pn
        sn_all[k, :k+1, ...] = S.sn_all
        sn[k, :k+1, ...] = S.sn

        if world.rank == 0:
            print "Duration (all orders):", solve_duration
            print "Number of measures (all procs, all orders):", S.nb_measures_all
            print

        if status == 2 : break # Received signal, terminate

    on_result = staircase_perturbation_series(c0, pn_all, sn_all, parameters['U'])

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = staircase_perturbation_series(c0, pn, sn, parameters['U'])
    on_error = variance_error(on, world)

    return np.squeeze(on_result), np.squeeze(on_error)

def save_staircase_solve(g0_lesser, g0_greater, _parameters, filename, kind_list, save_period=-1):

    if len(_parameters['measure_keldysh_indices']) != len(kind_list) :
        raise ValueError, 'kind_list has wrong shape'

    world = MPI.COMM_WORLD

    if world.rank == 0:
        print "\n----------- Save staircase solver -----------"

    parameters = _parameters.copy()
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    solve_duration = 0

    for k in range(1, max_order + 1):

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k

        S = SolverCore(**parameters)
        S.set_g0(g0_lesser, g0_greater)

        # order zero and initializing arrays
        if k == 1:
            c0, s0 = S.order_zero

            # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
            pn_all = np.empty((max_order+1, max_order+1), dtype=int) * np.nan
            pn = pn_all.copy()
            sn_all = np.empty(pn_all.shape + s0.shape, dtype=complex) * np.nan
            sn = sn_all.copy()

            pn_all[0, 0] = 1 # needed to avoid NaNs, any non zero value works
            pn[0, 0] = 1
            sn_all[0, 0, ...] = s0
            sn[0, 0, ...] = s0

        status = 1
        while status == 1:
            status = S.run(save_period)

            pn_all[k, :k+1] = S.pn_all
            pn[k, :k+1] = S.pn
            sn_all[k, :k+1, ...] = S.sn_all
            sn[k, :k+1, ...] = S.sn

            if world.rank == 0:
                print pn_all

            on_result = staircase_perturbation_series(c0, pn_all, sn_all, parameters['U'])

            # Calculate error
            # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
            on = staircase_perturbation_series(c0, pn, sn, parameters['U'])
            on_error = variance_error(on, world)

            if world.rank == 0:
                print "Duration (all orders):", solve_duration + S.solve_duration
                print "Number of measures (all procs, all orders):", S.nb_measures_all
                print
                save(solve_duration + S.solve_duration, S.nb_measures_all, parameters, on_result[:k+1], on_error[:k+1], filename, kind_list, world.size)

        solve_duration += S.solve_duration
        if status == 2: break # Received signal, terminate

    return np.squeeze(on_result), np.squeeze(on_error)

