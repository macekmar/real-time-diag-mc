# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series, staircase_perturbation_series_cum, compute_cn, staircase_leading_elts
import numpy as np
import itertools
from pytriqs.archive import HDFArchive
from datetime import datetime
from time import clock

def reverse_axis(array):
    """For first dimensions broadcasting"""
    return np.transpose(array, tuple(np.arange(array.ndim)[::-1]))

def variance_error(on, comm=MPI.COMM_WORLD, weight=None):
    weighted = weight is not None
    if comm.rank != 0:
        comm.gather(on)
        if weighted: comm.gather(weight)
        on_error = comm.bcast(None)
    else:
        on_all = np.array(comm.gather(on), dtype=complex)
        if weighted:
            weight_all = np.array(comm.gather(weight))
            weight_sum = np.sum(weight_all, axis=0)
            weight_all[:, weight_sum == 0] = 1 # do not consider weight which sums to 0

            # reshape
            weights = np.zeros(on_all.shape[::-1])
            weights[...] = reverse_axis(weight_all)
            weights = reverse_axis(weights)

        else:
            weights = None

        avg_on_all = np.average(on_all, weights=weights, axis=0)
        var_on_all = np.average(np.abs(on_all - avg_on_all[np.newaxis, ...])**2, weights=weights, axis=0)

        on_error = np.sqrt(var_on_all / comm.size)
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

def write(duration, nb_measures, parameters, on, on_error, filename, kind_list, nb_proc):
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

def write_add(data, data_name, filename):
    with HDFArchive(filename, 'a') as ar:
        ar[data_name] = data


def calc_ideal_U(pn, U):
    """calculates ideal U value"""
    nonzero_ind = np.nonzero(pn)[0]
    pn_nonzero = pn[nonzero_ind].astype(np.float32)
    if len(pn_nonzero) >= 2:
        power = float(nonzero_ind[-1] - nonzero_ind[-2])
        U_proposed = U * pow(2. * pn_nonzero[-2] / pn_nonzero[-1], 1. / power)
    else:
        U_proposed = U
    return U_proposed

class Results(dict):

    def __init__(self, world):
        super(Results, self).__init__()
        self.nb_runs = 0
        self.world = world
        self['pn'] = []
        self['pn_all'] = []
        self['sn'] = []
        self['sn_all'] = []
        self['kernels'] = []
        self['kernels_all'] = []
        self['U'] = []
        self['durations'] = []
        self['nb_measures'] = []

        self['bin_times'] = None

    def append_empty_slot(self):
        self.nb_runs += 1
        self['pn'].append(None)
        self['pn_all'].append(None)
        self['sn'].append(None)
        self['sn_all'].append(None)
        self['kernels'].append(None)
        self['kernels_all'].append(None)
        self['U'].append(None)
        self['durations'].append(None)
        self['nb_measures'].append(None)

    def fill_last_slot(self, solver_core):
        assert(len(self['pn']) == 1 or len(solver_core.pn) >= len(self['pn'][-2]))
        self['pn'][-1] = solver_core.pn
        self['pn_all'][-1] = solver_core.pn_all
        self['sn'][-1] = solver_core.sn
        self['sn_all'][-1] = solver_core.sn_all
        self['kernels'][-1] = solver_core.kernels
        self['kernels_all'][-1] = solver_core.kernels_all
        self['U'][-1] = solver_core.U
        self['durations'][-1] = solver_core.solve_duration
        self['nb_measures'][-1] = solver_core.nb_measures_all

        self['bin_times'] = solver_core.bin_times

    def compute_cn_on(self):
        assert(self.nb_runs > 0)
        assert('c0' in self)
        self['cn'] = compute_cn(self['pn'], self['U'], self['c0'])
        self['cn_all'] = compute_cn(self['pn_all'], self['U'], self['c0'])
        self['cn_error'] = variance_error(self['cn'], self.world)
        self['sn_error'] = []
        for pn, sn in zip(self['pn'], self['sn']):
            self['sn_error'].append(variance_error(sn, self.world, weight=pn))
        n_dim_sn = self['sn'][0][0].ndim
        slicing = (slice(None),) + n_dim_sn * (np.newaxis,) # for broadcasting cn array onto sn array
        self['on'] = staircase_leading_elts(self['sn']) * self['cn'][slicing]
        self['on_all'] = staircase_leading_elts(self['sn_all']) * self['cn_all'][slicing]
        self['on_error'] = variance_error(self['on'], self.world) #TODO: weight with nb of measures



class SolverPython(object):

    def __init__(self, g0_lesser, g0_greater, parameters, filename=None, kind_list=None):
        self.g0_lesser = g0_lesser
        self.g0_greater = g0_greater
        self.parameters = parameters.copy()
        self.world = MPI.COMM_WORLD
        self.res = Results(self.world)
        self.qmc_duration = 0.
        self.nb_measures_all = 0
        self.nb_measures = 0

        self.filename = filename
        self.kind_list = kind_list

    def order_zero(self):
        params = self.parameters.copy()
        params['U'] = float('NaN')
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        c0, s0 = S.order_zero
        self.res['c0'] = c0

    def checkpoint(self, solver_core, computes_on=False):
        if self.world.rank == 0:
            print datetime.now(), ": Order", solver_core.max_order
            print 'pn:', solver_core.pn_all
            print

        self.res.fill_last_slot(solver_core)
        self.res.compute_cn_on()

        if self.world.rank == 0:
            self.write(self.filename, self.kind_list)

    def prerun_and_run(self, nb_warmup_cycles, nb_cycles, order, U, save_period=-1):
        self.res.append_empty_slot()
        params = self.parameters.copy()

        params['max_perturbation_order'] = order
        params['U'] = U
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        # prerun
        start = clock()
        if self.world.rank == 0:
            print 'Pre run'
        S.run(nb_warmup_cycles, True)
        prerun_duration = float(clock() - start)
        prerun_duration = self.world.allreduce(prerun_duration) / float(self.world.size) # take average
        new_U = calc_ideal_U(S.pn_all, U)

        # time estimation
        if save_period > 0:
            save_period = max(save_period, 60) # 1 minute mini
            nb_cycles_per_subrun = int(float(nb_warmup_cycles * save_period) / prerun_duration + 0.5)
        else:
            nb_cycles_per_subrun = nb_cycles

        if self.world.rank == 0:
            print 'Nb cycles per subrun =', nb_cycles_per_subrun
            print 'Estimated run time =', prerun_duration * float(nb_cycles + nb_warmup_cycles) / float(nb_warmup_cycles), 'seconds'

        # prepare new solver taking prerun into account
        if self.world.rank == 0:
            print 'changing U:', U, '=>', new_U
        params['U'] = new_U
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        # warmup
        if self.world.rank == 0:
            print 'Warmup'
        S.run(nb_warmup_cycles, False)

        # main run
        if self.world.rank == 0:
            print 'main runs'
        while S.nb_measures < nb_cycles:
            nb_cycles_remaining = nb_cycles - S.nb_measures
            if nb_cycles_remaining > nb_cycles_per_subrun:
                S.run(nb_cycles_per_subrun, True)
                self.checkpoint(S)
            else: # last run
                S.run(nb_cycles_remaining, True)
                S.compute_sn_from_kernels
                self.checkpoint(S, computes_on=True)

        self.qmc_duration += S.solve_duration
        self.nb_measures_all += S.nb_measures_all
        self.nb_measures += S.nb_measures

    def write(self, filename, kind_list):
        # before using save, check kind_list has the good shape
        if filename is not None:
            with HDFArchive(filename, 'w') as ar:
                ar['nb_proc'] = self.world.size
                ar['interaction_start'] = self.parameters['interaction_start']
                ar['times'] = self.parameters['measure_times']
                ar['bin_times'] = np.array(self.res['bin_times'])

                if 'on_all' in self.res:
                    for i, kind in enumerate(kind_list):
                        ar.create_group(kind)
                        group = ar[kind]
                        group['times'] = self.parameters['measure_times'] # for backward compatibility
                        group['on'] = np.squeeze(self.res['on_all'][:, :, i])
                        group['on_error'] = np.squeeze(self.res['on_error'][:, :, i])

                for key in ['pn_all', 'cn_all', 'sn_all', 'kernels_all']:
                    ar[key[:-4]] = self.res[key]

                for key in ['cn_error', 'sn_error']:
                    ar[key] = self.res[key]

                for key in ['durations', 'nb_measures']:
                    ar[key] = np.array(self.res[key])
                ar['run_time'] = np.sum(self.res['durations'])

            print 'Saved in', filename
        else:
            return


######################### Front End #######################

def single_solve(g0_lesser, g0_greater, parameters_, filename=None, kind_list=None, save_period=-1, histogram=False):
    if histogram:
        raise NotImplementedError

    parameters = parameters_.copy()
    nb_warmup_cycles = parameters.pop('n_warmup_cycles')
    nb_cycles = parameters.pop('n_cycles')

    solver = SolverPython(g0_lesser, g0_greater, parameters, filename, kind_list)
    solver.order_zero()
    solver.prerun_and_run(nb_warmup_cycles, nb_cycles, parameters['max_perturbation_order'],
                          parameters['U'], save_period=save_period)

    return np.squeeze(solver.res['on_all']), np.squeeze(solver.res['on_error'])

def single_solve_old(g0_lesser, g0_greater, parameters, filename=None, kind_list=None, save_period=-1, histogram=False):
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
            write(S.solve_duration, S.nb_measures_all, parameters, on, on_error, filename, kind_list, world.size)
            write_add(S.kernels_all, 'kernels', filename)

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

def staircase_solve(g0_lesser, g0_greater, parameters_, filename=None, kind_list=None, save_period=-1, only_even=False):
    parameters = parameters_.copy()
    nb_warmup_cycles = parameters.pop('n_warmup_cycles')
    nb_cycles = parameters.pop('n_cycles')
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    U_list = parameters['U']

    if not only_even:
        orders = list(range(1, max_order + 1))
    else:
        orders = list(range(2, max_order + 1, 2))

    solver = SolverPython(g0_lesser, g0_greater, parameters, filename, kind_list)
    solver.order_zero()

    for i, k in enumerate(orders):
        if solver.world.rank == 0:
            print "---------------- Order", k, "----------------"
        solver.prerun_and_run(nb_warmup_cycles, nb_cycles, k, U_list[i], save_period=save_period)

    return np.squeeze(solver.res['on_all']), np.squeeze(solver.res['on_error'])

def staircase_solve_old(g0_lesser, g0_greater, _parameters, filename=None, kind_list=None, save_period=-1, only_even=False):
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

    U_list = parameters["U"]
    if len(U_list) < len(orders):
        raise RuntimeError, 'Number of U must match number of non zero orders'
    parameters["U"] = U_list[0]

    pn_all = (len(orders) + 1) * [None]
    pn     = (len(orders) + 1) * [None]
    sn_all = (len(orders) + 1) * [None]
    sn     = (len(orders) + 1) * [None]
    kernels = (max_order + 1) * [None]
    nb_kernels = (max_order + 1) * [None]

    # order zero
    S = SolverCore(**parameters)
    S.set_g0(g0_lesser, g0_greater)
    c0, s0 = S.order_zero

    pn_all[0] = [1]
    pn[0] = [1]
    sn_all[0] = s0[np.newaxis, :]
    sn[0] = s0[np.newaxis, :]

    k_catchup = 0
    for i, k in enumerate(orders):
        i += 1 # i=0 is reserved for order zero

        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        parameters["max_perturbation_order"] = k
        parameters["U"] = U_list[i - 1]

        S = SolverCore(**parameters)
        S.set_g0(g0_lesser, g0_greater)

        abort = False
        end_subrun = False
        while not (abort or end_subrun):
            status = S.run(save_period)

            status_all = world.allgather(status)
            abort = 2 in status_all
            end_subrun = 1 not in status_all # subrun ends when all proc are done with it

            pn_all[i] = S.pn_all
            pn[i] = S.pn
            sn_all[i] = S.sn_all
            sn[i] = S.sn
            while k_catchup <= k:
                kernels[k_catchup] = S.kernels_all[k_catchup, ...]
                nb_kernels[k_catchup] = S.nb_kernels[k_catchup, ...]
                k_catchup += 1

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
                U_proposed = parameters['U'] * pow(2. * pn_nonzero[-2] / pn_nonzero[-1], 1. / power)
            else:
                U_proposed = None

            # calculates results
            on_result, cn_result = staircase_perturbation_series(c0, pn_all[:i+1], sn_all[:i+1], U_list)

            # Calculate error
            # estimation using the variance of the On obtained on each process (assumes each process is equivalent)
            on, cn = staircase_perturbation_series(c0, pn[:i+1], sn[:i+1], U_list)
            on_error = variance_error(on, world)

            # print "Number of measures (procs", world.rank, ", this order):", S.nb_measures

            if world.rank == 0:
                print datetime.now(), ": Order", k
                print "Duration (this order):", S.solve_duration
                print "Duration (all orders):", solve_duration + S.solve_duration
                print "Number of measures (all procs, this order):", S.nb_measures_all
                print "Number of measures (all procs, all orders):", nb_measures + S.nb_measures_all
                print "U proposed:", U_proposed, "(current is", parameters['U'], ")"
                print
                if save:
                    write(solve_duration + S.solve_duration, nb_measures + S.nb_measures_all, parameters, on_result[:k+1], on_error[:k+1], filename, kind_list, world.size)
                    write_add(np.array(kernels[:k+1], dtype=complex), 'kernels', filename)
                    write_add(np.array(nb_kernels[:k+1], dtype=int), 'nb_kernels', filename)
                    write_add(np.array(cn_result), 'cn', filename)

        solve_duration += S.solve_duration
        nb_measures += S.nb_measures_all
        if abort: break # Received signal, terminate

    del S

    return np.squeeze(on_result), np.squeeze(on_error)

if __name__ == '__main__':
    world = MPI.COMM_WORLD

    if world.size != 4:
        raise RuntimeError('These tests must be run with MPI on 4 processes')

    # test variance_error
    if world.rank == 0:
        array = np.array([12.3, 68., 0.52])
    elif world.rank == 1:
        array = np.array([4.5, -6.3, 9.4])
    elif world.rank == 2:
        array = np.array([-86.4, 40, 63.1])
    elif world.rank == 3:
        array = np.array([10, 20, 30])

    error_ref = [20.688855695760459, 13.603693202582892, 12.033703451140882]
    error = variance_error(array, world)
    assert(error.shape == (3,))
    assert(np.allclose(error, error_ref, atol=1e-5))

    # test variance_error
    if world.rank == 0:
        array = np.array([[12.3, 0.], [68., 0.], [0.52, 1]])
    elif world.rank == 1:
        array = np.array([[4.5, 1.], [-6.3, 0], [9.4, 1]])
    elif world.rank == 2:
        array = np.array([[-86.4, 0.], [40, 0.], [63.1, 0]])
    elif world.rank == 3:
        array = np.array([[10, 0.], [20, 0], [30, 0]])

    error_ref = [[20.688855695760459, 0.21650635094610965], [13.603693202582892, 0.], [12.033703451140882, 0.25]]
    error = variance_error(array, world)
    assert(error.shape == (3, 2))
    assert(np.allclose(error, error_ref, atol=1e-5))

    # test variance_error
    if world.rank == 0:
        array = np.array([[12.3 +0j, 0.], [68., 0.], [0.52, 1]])
        weight = np.array([100, 100, 0])
    elif world.rank == 1:
        array = np.array([[4.5, 1.], [-6.3, 0], [9.4, 1]])
        weight = np.array([75., 75., 0])
    elif world.rank == 2:
        array = np.array([[-86.4, 0.], [40, 0.], [63.1, 0]])
        weight = np.array([0, 0, 0])
    elif world.rank == 3:
        array = np.array([[10, 0.], [20, 0], [30, 0]])
        weight = np.array([60, 60, 0])

    error_ref = [[1.6809385918497846, 0.2330734287256026], [16.251235980435649, 0.], [12.033703451140882, 0.25]]
    error = variance_error(array, world, weight=weight)
    # print error
    assert(error.shape == (3, 2))
    assert(np.allclose(error, error_ref, atol=1e-5))

