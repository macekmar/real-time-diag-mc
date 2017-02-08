# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series
import numpy as np


def variance_error(on, comm):

    if comm.rank != 0:
        comm.gather(on)
        on_error = comm.bcast(None)
    else:
        on_all = np.array(comm.gather(on), dtype=complex)

        print on_all
        on_error = np.sqrt(np.var(on_all, axis=0) / comm.size)
        comm.bcast(on_error)

    return on_error


def single_solve(g0_lesser, g0_greater, parameters):

    world = MPI.COMM_WORLD
    na = np.newaxis

    S = SolverCore(g0_lesser, g0_greater)
    (pn, sn), (pn_error, sn_error) = S.solve(**parameters)
    nb_measures = S.nb_measures

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
        world.reduce(sn * pn[:, na] * nb_measures, op=MPI.SUM)

        on_result = world.bcast(None)

    else:
        nb_measures_all = world.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = world.reduce(pn * nb_measures, op=MPI.SUM)
        pn_avg = nb_measures_by_order / nb_measures_all
        sn_avg = world.reduce(sn * pn[:, na] * nb_measures, op=MPI.SUM) / nb_measures_by_order[:, na]

        on_result = perturbation_series(c0, pn_avg, sn_avg, U)

        world.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = perturbation_series(c0, pn, sn, U)
    
    on_error = variance_error(on, world)

    return on_result, on_error


def staircase_solve(g0_lesser, g0_greater, _parameters):

    world = MPI.COMM_WORLD
    na = np.newaxis

    if world.rank == 0:
        print "----------- Staircase solver -----------"

    S = SolverCore(g0_lesser, g0_greater)

    parameters = _parameters.copy()
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    U = parameters["U"]

    # pn_all and sn_all are lower triangular arrays (for 2 first dims), unused values are set to NaN.
    pn_all = np.empty((max_order+1, max_order+1)) * np.nan
    sn_all = np.empty((max_order+1, max_order+1, len(parameters["measure_times"])), dtype=complex) * np.nan
    nb_measures = np.zeros((max_order+1,), dtype=int)

    for k in range(max_order + 1):

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k
        (pn, sn), _ = S.solve(**parameters)

        if k == 0:
            c0 = pn
        else:
            pn_all[k, :k+1] = pn
        sn_all[k, :k+1, :] = sn
        if k == 0:
            nb_measures[0] = 1 # needed to avoid divide by zero, any value works
        else:
            nb_measures[k] = S.nb_measures

    # Calculate estimation
    if world.rank != 0:
        world.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = pn_all * nb_measures[:, na]
        world.reduce(nb_measures_by_order, op=MPI.SUM)
        world.reduce(sn_all * nb_measures_by_order[:, :, na], op=MPI.SUM)

        on_result = world.bcast(None)

    else:
        nb_measures_all = world.reduce(nb_measures, op=MPI.SUM)
        nb_measures_by_order = pn_all * nb_measures[:, na]
        nb_measures_by_order = world.reduce(nb_measures_by_order, op=MPI.SUM)
        pn_avg = nb_measures_by_order / nb_measures_all[:, na]
        sn_avg = world.reduce(sn_all * nb_measures_by_order[:, :, na], op=MPI.SUM) / nb_measures_by_order[:, :, na]

        on_result = staircase_perturbation_series(c0, pn_avg, sn_avg, U)
    
        world.bcast(on_result)

    # Calculate error
    # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
    on = staircase_perturbation_series(c0, pn_all, sn_all, U)

    on_error = variance_error(on, world)

    return on_result, on_error

