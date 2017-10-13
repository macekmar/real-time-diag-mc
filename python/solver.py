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


# def calc_ideal_U(pn, U):
#     """calculates ideal U value"""
#     nonzero_ind = np.nonzero(pn)[0]
#     pn_nonzero = pn[nonzero_ind].astype(np.float32)
#     if len(pn_nonzero) >= 2:
#         power = float(nonzero_ind[-1] - nonzero_ind[-2])
#         U_proposed = U * pow(2. * pn_nonzero[-2] / pn_nonzero[-1], 1. / power)
#     else:
#         U_proposed = U
#     return U_proposed

from scipy.stats import linregress
def calc_ideal_U(pn, U):
    """calculates ideal U value"""
    if len(pn) < 2:
        raise ValueError('pn must be at least of length 2')
    an = [max(pn[0], pn[1])]
    for k in range(1, len(pn)-1):
        an.append(0.5*(max(pn[k-1], pn[k]) + max(pn[k], pn[k+1])))
    an.append(max(pn[-2], pn[-1]))
    slope, _, _, _, _ = linregress(np.arange(len(an)), np.log(an))
    return 2. * U * np.exp(-slope)

class Results(dict):

    def __init__(self, world, starttime):
        super(Results, self).__init__()
        self.nb_runs = 0
        self.world = world
        self.starttime = starttime
        self['pn'] = []
        self['pn_all'] = []
        self['sn'] = []
        self['sn_all'] = []
        self['kernels'] = []
        self['kernels_all'] = []
        self['nb_kernels'] = []
        self['nb_kernels_all'] = []
        self['U'] = []
        self['durations'] = []
        self['nb_measures'] = []

        self['bin_times'] = None
        self['run_time'] = None

    def append_empty_slot(self):
        self.nb_runs += 1
        self['pn'].append(None)
        self['pn_all'].append(None)
        self['sn'].append(None)
        self['sn_all'].append(None)
        self['kernels'].append(None)
        self['kernels_all'].append(None)
        self['nb_kernels'].append(None)
        self['nb_kernels_all'].append(None)
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
        self['nb_kernels'][-1] = solver_core.nb_kernels
        self['nb_kernels_all'][-1] = solver_core.nb_kernels_all
        self['U'][-1] = solver_core.U
        self['durations'][-1] = solver_core.solve_duration
        self['nb_measures'][-1] = solver_core.nb_measures_all

        self['bin_times'] = np.array(solver_core.bin_times)
        self['run_time'] = (datetime.now() - self.starttime).total_seconds()

    def compute_cn(self):
        assert(self.nb_runs > 0)
        assert('c0' in self)
        self['cn'] = compute_cn(self['pn'], self['U'], self['c0'])
        self['cn_all'] = compute_cn(self['pn_all'], self['U'], self['c0'])
        self['cn_error'] = variance_error(self['cn'], self.world)

    def compute_on(self):
        assert(self.nb_runs > 0)

        # sn
        self['sn_error'] = []
        for pn, sn in zip(self['pn'], self['sn']):
            self['sn_error'].append(variance_error(sn, self.world, weight=pn))

        # on
        n_dim_sn = self['sn'][0][0].ndim
        slicing = (slice(None),) + n_dim_sn * (np.newaxis,) # for broadcasting cn array onto sn array
        self['on'] = np.squeeze(staircase_leading_elts(self['sn']) * self['cn'][slicing])
        self['on_all'] = np.squeeze(staircase_leading_elts(self['sn_all']) * self['cn_all'][slicing])
        self['on_error'] = np.squeeze(variance_error(self['on'], self.world)) #TODO: weight with nb of measures


    def save(self, filename, params):
        with HDFArchive(filename, 'w') as ar:
            ar['nb_proc'] = self.world.size
            ar['interaction_start'] = params['interaction_start']
            ar['times'] = params['measure_times']

            for key in self:
                if key[-4:] == '_all':
                    ar[key[:-4]] = self[key]
                elif key + '_all' in self:
                    pass
                else:
                    ar[key] = self[key]

        print 'Saved in', filename


class SolverPython(object):

    def __init__(self, g0_lesser, g0_greater, parameters, filename=None, kind_list=None):
        self.g0_lesser = g0_lesser
        self.g0_greater = g0_greater
        self.parameters = parameters.copy()
        self.world = MPI.COMM_WORLD
        self.starttime = datetime.now()
        self.res = Results(self.world, self.starttime)

        self.filename = filename
        self.kind_list = kind_list

    def order_zero(self):
        params = self.parameters.copy()
        params['U'] = float('NaN')
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        c0, s0 = S.order_zero
        self.res['c0'] = c0

    def checkpoint(self, solver_core, compute_on=False):
        if self.world.rank == 0:
            print datetime.now(), ": Order", solver_core.max_order
            print 'pn:', solver_core.pn_all
            print

        self.res.fill_last_slot(solver_core)
        self.res.compute_cn()
        if compute_on:
            self.res.compute_on()

        if self.world.rank == 0 and self.filename is not None:
            self.res.save(self.filename, self.parameters)

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
            print 'Changing U:', U, '=>', new_U
        params['U'] = new_U
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        # warmup
        if self.world.rank == 0:
            print 'Warmup'
        S.run(nb_warmup_cycles, False)

        # main run
        if self.world.rank == 0:
            print 'Main runs'
        while S.nb_measures < nb_cycles:
            nb_cycles_remaining = nb_cycles - S.nb_measures
            if nb_cycles_remaining > nb_cycles_per_subrun:
                S.run(nb_cycles_per_subrun, True)
                self.checkpoint(S)
            else: # last run
                S.run(nb_cycles_remaining, True)
                S.compute_sn_from_kernels
                self.checkpoint(S, compute_on=True)


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

    if len(U_list) != len(orders):
        raise ValueError, "Size of list of U must match the number of orders to compute"

    solver = SolverPython(g0_lesser, g0_greater, parameters, filename, kind_list)
    solver.order_zero()

    for i, k in enumerate(orders):
        if solver.world.rank == 0:
            print "---------------- Order", k, "----------------"
        solver.prerun_and_run(nb_warmup_cycles, nb_cycles, k, U_list[i], save_period=save_period)

    return np.squeeze(solver.res['on_all']), np.squeeze(solver.res['on_error'])


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

