# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series
import numpy as np
import itertools


def variance_error(on, comm):

    if comm.rank != 0:
        comm.gather(on)
        on_error = comm.bcast(None)
    else:
        on_all = np.array(comm.gather(on), dtype=complex)

        on_error = np.sqrt(np.var(on_all, axis=0) / comm.size)
        comm.bcast(on_error)

    return on_error


def single_solve(g0_lesser, g0_greater, parameters, histogram=False):

    world = MPI.COMM_WORLD

    S = SolverCore(g0_lesser, g0_greater)
    (pn, sn), (pn_error, sn_error) = S.solve(**parameters)
    nb_measures = S.nb_measures
    list_config = S.config_list
    weight_config = S.config_weight
    list_config = map(lambda x, y : [x] + list(y), weight_config, list_config)

    # make pn shape broadcastable with sn
    while pn.ndim < sn.ndim:
        pn = pn[..., np.newaxis]

    U = parameters["U"]

    # Calculate c0
    parameters_0 = parameters.copy()
    parameters_0["max_perturbation_order"] = 0
    parameters_0["min_perturbation_order"] = 0
    (p0, s0), _ = S.solve(**parameters_0)
    c0 = p0

    # Calculate estimation
    if world.rank != 0:
        world.reduce(nb_measures, op=MPI.SUM)
        world.reduce(pn * nb_measures, op=MPI.SUM)
        world.reduce(sn * pn * nb_measures, op=MPI.SUM)

        on_result = world.bcast(None)

    else:
        nb_measures_all = world.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = world.reduce(pn * nb_measures, op=MPI.SUM)
        pn_avg = nb_measures_by_order / nb_measures_all
        sn_avg = world.reduce(sn * pn * nb_measures, op=MPI.SUM) / nb_measures_by_order

        on_result = perturbation_series(c0, np.squeeze(pn_avg), sn_avg, U)

        world.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = perturbation_series(c0, np.squeeze(pn), sn, U)
    
    on_error = variance_error(on, world)
    output = (on_result, on_error)

    # Sort out the configurations if histogram is True
    if histogram:
        list_config.sort(key=len)
        list_histograms = []
        for k, g in itertools.groupby(list_config, len):
            list_histograms.append(np.array(list(g)).T)

        list_histograms_all = []

        for histogram in list_histograms:
            if world.rank != 0:
                world.gather(histogram)
            else:
                list_histograms_all.append(np.concatenate(world.gather(histogram), axis=1))

        if world.rank != 0:
            list_histograms_all = world.bcast(None)
        else:
            world.bcast(list_histograms_all)
        output += (list_histograms_all,)

    return output


def staircase_solve(g0_lesser, g0_greater, _parameters):

    world = MPI.COMM_WORLD
    na = np.newaxis

    if world.rank == 0:
        print "\n----------- Staircase solver -----------"

    S = SolverCore(g0_lesser, g0_greater)

    parameters = _parameters.copy()
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    U = parameters["U"]

    for k in range(max_order + 1):

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k
        (pn, sn), _ = S.solve(**parameters)

        if k == 0:
            
            # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
            pn_all = np.empty((max_order+1, max_order+1)) * np.nan
            sn_all = np.empty(pn_all.shape + sn.shape[1:], dtype=complex) * np.nan
            nb_measures = np.zeros((max_order+1,), dtype=int)

            c0 = pn
            pn_all[0, 0] = 1. # needed to avoid NaNs, any value works
            nb_measures[0] = 1 # needed to avoid divide by zero, any value works
        else:
            pn_all[k, :k+1] = pn
            nb_measures[k] = S.nb_measures

        sn_all[k, :k+1, ...] = sn

        if world.rank != 0:
            world.reduce(S.nb_measures)
        else:
            tot_nb_measures = world.reduce(S.nb_measures)
            print "[Solver] Total number of measures:", tot_nb_measures

    while pn_all.ndim < sn_all.ndim:
        pn_all = pn_all[..., na]
    while nb_measures.ndim < sn_all.ndim:
        nb_measures = nb_measures[..., na]

    # Calculate estimation
    if world.rank != 0:
        world.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = pn_all * nb_measures
        world.reduce(nb_measures_by_order, op=MPI.SUM)
        world.reduce(sn_all * nb_measures_by_order, op=MPI.SUM)

        on_result = world.bcast(None)

    else:
        nb_measures_all = world.reduce(nb_measures, op=MPI.SUM)
        print "[Solver] Total number of measures (all orders):", nb_measures_all[1:].sum()
        nb_measures_by_order = pn_all * nb_measures
        nb_measures_by_order_all = world.reduce(nb_measures_by_order, op=MPI.SUM)
        pn_avg = nb_measures_by_order_all / nb_measures_all
        sn_avg = world.reduce(sn_all * nb_measures_by_order, op=MPI.SUM) / nb_measures_by_order_all

        on_result = staircase_perturbation_series(c0, np.squeeze(pn_avg), sn_avg, U)

        world.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = staircase_perturbation_series(c0, np.squeeze(pn_all), sn_all, U)

    on_error = variance_error(on, world)

    return on_result, on_error

