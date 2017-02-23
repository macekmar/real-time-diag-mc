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

    # Sort out the configurations if histogram is True
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

def gather_on(pn, sn, nb_measures, c0, U, comm=MPI.COMM_WORLD):

    # make pn shape broadcastable with sn
    while pn.ndim < sn.ndim:
        pn = pn[..., np.newaxis]

    # Calculate estimation
    if comm.rank != 0:
        comm.reduce(nb_measures, op=MPI.SUM)
        comm.reduce(pn * nb_measures, op=MPI.SUM)
        comm.reduce(sn * pn * nb_measures, op=MPI.SUM)

        on_result = comm.bcast(None)

    else:
        nb_measures_all = comm.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = comm.reduce(pn * nb_measures, op=MPI.SUM)
        pn_avg = nb_measures_by_order / nb_measures_all
        sn_avg = comm.reduce(sn * pn * nb_measures, op=MPI.SUM) / nb_measures_by_order

        on_result = perturbation_series(c0, np.squeeze(pn_avg), sn_avg, U)

        comm.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = perturbation_series(c0, np.squeeze(pn), sn, U)
    on_error = variance_error(on, comm)

    return on_result, on_error

def staircase_gather_on(pn_all, sn_all, nb_measures, c0, U, comm=MPI.COMM_WORLD):
    na = np.newaxis

    while pn_all.ndim < sn_all.ndim:
        pn_all = pn_all[..., na]
    while nb_measures.ndim < sn_all.ndim:
        nb_measures = nb_measures[..., na]

    # Calculate estimation
    if comm.rank != 0:
        comm.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = pn_all * nb_measures
        comm.reduce(nb_measures_by_order, op=MPI.SUM)
        comm.reduce(sn_all * nb_measures_by_order, op=MPI.SUM)

        on_result = comm.bcast(None)

    else:
        nb_measures_all = comm.reduce(nb_measures, op=MPI.SUM)
        print "[Solver] Total number of measures (all orders):", nb_measures_all[1:].sum()
        nb_measures_by_order = pn_all * nb_measures
        nb_measures_by_order_all = comm.reduce(nb_measures_by_order, op=MPI.SUM)
        pn_avg = nb_measures_by_order_all / nb_measures_all
        sn_avg = comm.reduce(sn_all * nb_measures_by_order, op=MPI.SUM) / nb_measures_by_order_all

        on_result = staircase_perturbation_series(c0, np.squeeze(pn_avg), sn_avg, U)

        comm.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = staircase_perturbation_series(c0, np.squeeze(pn_all), sn_all, U)
    on_error = variance_error(on, comm)

    return on_result, on_error

def save(solver, parameters, on, on_error, filename, kind_list, nb_proc):
    # before using save, check kind_list has the good shape
    with HDFArchive(filename, 'w') as ar:
        ar['nb_proc'] = nb_proc
        ar['interaction_start'] = parameters['interaction_start']
        ar['run_time'] = solver.solve_duration
        ar['nb_measures'] = solver.nb_measures

        for i, kind in enumerate(kind_list):
            ar.create_group(kind)
            group = ar[kind]
            group['times'] = parameters['measure_times']
            group['on'] = np.squeeze(on[:, :, i])
            group['on_error'] = np.squeeze(on_error[:, :, i])

    print 'Saved in', filename


######################### Front End #######################

def single_solve(g0_lesser, g0_greater, parameters, max_time=-1, histogram=False):

    world = MPI.COMM_WORLD

    S = SolverCore(**parameters)
    S.set_g0(g0_lesser, g0_greater)

    S.order_zero
    c0 = S.pn[0]

    S.run(max_time)
    on, on_error = gather_on(S.pn, S.sn, S.nb_measures, c0, parameters['U'], world)
    output = (np.squeeze(on), np.squeeze(on_error))
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

    S.order_zero
    c0 = S.pn[0]

    status = 1
    while status == 1:
        status = S.run(save_period) # status=0 if finished, 1 if timed up, 2 if aborted
        on, on_error = gather_on(S.pn, S.sn, S.nb_measures, c0, parameters['U'], world)

        if world.rank == 0:
            save(S, parameters, on, on_error, filename, kind_list, world.size)

        output = (np.squeeze(on), np.squeeze(on_error))

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
            S.order_zero
            pn = S.pn
            sn = S.sn

            # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
            pn_all = np.empty((max_order+1, max_order+1)) * np.nan
            sn_all = np.empty(pn_all.shape + sn.shape[1:], dtype=complex) * np.nan
            nb_measures = np.zeros((max_order+1,), dtype=int)

            c0 = pn[0]
            pn_all[0, 0] = 1. # needed to avoid NaNs, any value works
            sn_all[0, 0, ...] = sn[0, ...]
            nb_measures[0] = 1 # needed to avoid divide by zero, any value works

        status = S.run(max_time - solve_duration)
        pn = S.pn
        sn = S.sn
        solve_duration += S.solve_duration

        pn_all[k, :k+1] = pn
        sn_all[k, :k+1, ...] = sn
        nb_measures[k] = S.nb_measures

        if world.rank != 0:
            world.reduce(S.nb_measures)
        else:
            tot_nb_measures = world.reduce(S.nb_measures)
            print "[Solver] Total number of measures:", tot_nb_measures

        if status == 2 : break # Received signal, terminate

    on_result, on_error = staircase_gather_on(pn_all, sn_all, nb_measures, c0, parameters["U"], world)

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

        if k == 1:
            # order zero and initializing arrays
            S.order_zero
            pn = S.pn
            sn = S.sn

            # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
            pn_all = np.empty((max_order+1, max_order+1)) * np.nan
            sn_all = np.empty(pn_all.shape + sn.shape[1:], dtype=complex) * np.nan
            nb_measures = np.zeros((max_order+1,), dtype=int)

            c0 = pn[0]
            pn_all[0, 0] = 1. # needed to avoid NaNs, any value works
            sn_all[0, 0, ...] = sn[0, ...]
            nb_measures[0] = 1 # needed to avoid divide by zero, any value works

        status = 1
        while status == 1:
            status = S.run(save_period)
            pn = S.pn
            sn = S.sn
            solve_duration += S.solve_duration

            pn_all[k, :k+1] = pn
            sn_all[k, :k+1, ...] = sn
            nb_measures[k] = S.nb_measures

            if world.rank != 0:
                world.reduce(S.nb_measures)
            else:
                tot_nb_measures = world.reduce(S.nb_measures)
                print "[Solver] Total number of measures:", tot_nb_measures

            on_result, on_error = staircase_gather_on(pn_all, sn_all, nb_measures, c0, parameters["U"], world)
            if world.rank == 0:
                save(S, parameters, on_result, on_error, filename, kind_list, world.size)

        if status == 2: break # Received signal, terminate

    return np.squeeze(on_result), np.squeeze(on_error)

