# from pytriqs.utility import mpi
from mpi4py import MPI
from ctint_keldysh import SolverCore
from perturbation_series import perturbation_series, staircase_perturbation_series, staircase_perturbation_series_cum, compute_cn, staircase_leading_elts
import numpy as np
import itertools
from pytriqs.archive import HDFArchive
from datetime import datetime, timedelta
from time import clock

def reverse_axis(array):
    """For first dimensions broadcasting"""
    return np.transpose(array, tuple(np.arange(array.ndim)[::-1]))

def variance_error(on, comm=MPI.COMM_WORLD, weight=None):
    """
    Compute the error on `on` through sqrt(var(on)/(nb of proccess-1)), where the variance is calculated over
    the processes of communicator `comm`.
    `weight` can be a number (the weight of this process), an array with same shape as `on`, or None (same
    weights for all).
    Returns an array of complex with same shape as `on`.
    """
    if comm.size == 1:
        return np.zeros_like(on)

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

            # do not consider weight which sums to 0
            if (weight_all.ndim == 1):
                if (weight_sum == 0): weight_all = None
            else:
                weight_all[:, weight_sum == 0] = 1

        else:
            weight_all = None

        avg_on_all = np.average(on_all, weights=weight_all, axis=0)
        var_on_all = np.average(np.abs(on_all - avg_on_all[np.newaxis, ...])**2, weights=weight_all, axis=0)

        on_error = np.sqrt(var_on_all / float(comm.size - 1))
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

def reduce_binning(x, chunk_size, axis=-1):
    x = np.swapaxes(x, axis, 0)
    shape = (-1, chunk_size) + x.shape[1:]
    x = x[:x.shape[0] - x.shape[0] % chunk_size].reshape(shape).sum(axis=1)
    return np.swapaxes(x, axis, 0)

class Results(dict):

    def __init__(self, world, starttime, size_partition, nb_bins_sum):
        super(Results, self).__init__()
        self.nb_runs = 0
        self.world = world
        self.size_part = min(size_partition, world.size)
        color = world.rank // (world.size // self.size_part)
        # self.part = world.Split(color, world.rank) # split in at least `self.size_part` parts of equal size
        self.nb_bins_sum = nb_bins_sum
        self.is_part_master = (world.rank % (world.size // self.size_part) == 0) and (color < self.size_part)
        self.part_comm = world.Split(self.is_part_master, world.rank)
        if not self.is_part_master:
            self.part_comm.Free()
        else:
            assert self.part_comm.size == self.size_part

        self.starttime = starttime
        self['pn'] = []
        self['pn_each'] = []
        self['pn_part'] = []
        self['sn'] = []
        self['sn_each'] = []
        self['kernels'] = []
        self['kernel_diracs'] = []
        self['nb_kernels'] = []
        self['U'] = []
        self['durations'] = []
        self['nb_measures'] = []

        self['bin_times'] = None
        self['dirac_times'] = None
        self['run_time'] = None

        if self.world.rank == 0:
            self['pn_part_save'] = []
            self['kernels_part_save'] = []
            self['kernel_diracs_part_save'] = []

    def append_empty_slot(self):
        self.nb_runs += 1
        self['pn'].append(None)
        self['pn_each'].append(None)
        self['pn_part'].append(None)
        self['sn'].append(None)
        self['sn_each'].append(None)
        self['kernels'].append(None)
        self['kernel_diracs'].append(None)
        self['nb_kernels'].append(None)
        self['U'].append(None)
        self['durations'].append(None)
        self['nb_measures'].append(None)

        if self.world.rank == 0:
            self['pn_part_save'].append(None)
            self['kernels_part_save'].append(None)
            self['kernel_diracs_part_save'].append(None)

    def fill_last_slot(self, solver_core, compute_on=False):
        assert(len(self['pn']) == 1 or len(solver_core.pn) >= len(self['pn'][-2]))
        assert(self.nb_runs > 0)
        assert('c0' in self)
        self['pn'][-1] = solver_core.pn
        self['sn'][-1] = solver_core.sn
        self['kernels'][-1] = solver_core.kernels
        self['kernel_diracs'][-1] = solver_core.kernel_diracs
        self['nb_kernels'][-1] = solver_core.nb_kernels
        self['U'][-1] = solver_core.U
        self['durations'][-1] = solver_core.qmc_duration
        self['nb_measures'][-1] = solver_core.nb_measures

        self['bin_times'] = np.array(solver_core.bin_times)
        self['dirac_times'] = np.array(solver_core.dirac_times)
        self['run_time'] = (datetime.now() - self.starttime).total_seconds()
        self['cn'] = compute_cn(self['pn'], self['U'], self['c0'])

        solver_core.collect_results(self.world.size)
        self['pn_each'][-1] = solver_core.pn
        self['sn_each'][-1] = solver_core.sn

        solver_core.collect_results(self.size_part)
        self['pn_part'][-1] = solver_core.pn
        cn_part = compute_cn(self['pn_part'], self['U'], self['c0'])
        kernels_part = solver_core.kernels
        kernels_part = reduce_binning(kernels_part, self.nb_bins_sum, axis=1) / float(self.nb_bins_sum)
        self['bin_times_part'] = reduce_binning(self['bin_times'], self.nb_bins_sum) / float(self.nb_bins_sum)
        kernel_diracs_part = solver_core.kernel_diracs

        if self.is_part_master:
            pn_part = self.part_comm.gather(self['pn_part'][-1], 0)
            cn_part = self.part_comm.gather(cn_part, 0)
            kernels_part = self.part_comm.gather(kernels_part, 0)
            kernel_diracs_part = self.part_comm.gather(kernel_diracs_part, 0)

        if self.world.rank == 0:
            change_axis = lambda x: np.rollaxis(x, 0, x.ndim) # send first axis to last position
            self['pn_part_save'][-1] = change_axis(np.array(pn_part, dtype=int))
            self['cn_part_save'] = change_axis(np.array(cn_part, dtype=float))
            self['kernels_part_save'][-1] = change_axis(np.array(kernels_part, dtype=complex))
            self['kernel_diracs_part_save'][-1] = change_axis(np.array(kernel_diracs_part, dtype=complex))

        if compute_on:
            # values
            n_dim_sn = self['sn'][0][0].ndim
            slicing = (slice(None),) + n_dim_sn * (np.newaxis,) # for broadcasting cn array onto sn array
            self['on'] = np.squeeze(staircase_leading_elts(self['sn']) * self['cn'][slicing])

            # errors
            cn_each = compute_cn(self['pn_each'], self['U'], self['c0'])
            self['cn_error'] = variance_error(cn_each, self.world)
            self['sn_error'] = []
            sn_each = staircase_leading_elts(self['sn_each'])
            for pn, sn in zip(staircase_leading_elts(self['pn_each']), sn_each):
                self['sn_error'].append(variance_error(sn, self.world, weight=pn))
            self['sn_error'] = np.array(self['sn_error'])
            on_each = np.squeeze(sn_each * cn_each[slicing])
            self['on_error'] = np.squeeze(variance_error(on_each, self.world)) #TODO: weight with nb of measures

        solver_core.collect_results(1) # go back to collecting over world

    def save(self, filename, params):
        assert self.world.rank == 0
        with HDFArchive(filename, 'w') as ar:
            ar.create_group('metadata')
            metadata = ar['metadata']
            metadata['U'] = self['U']
            metadata['durations'] = self['durations']
            metadata['nb_proc'] = self.world.size
            metadata['interaction_start'] = params['interaction_start']
            metadata['nb_measures'] = self['nb_measures'] # cumulated over proc
            metadata['run_time'] = self['run_time']

            ar.create_group('kernels')
            kernels = ar['kernels']
            kernels['kernels'] = staircase_leading_elts(self['kernels'])
            kernels['nb_kernels'] = staircase_leading_elts(self['nb_kernels'])
            kernels['cn'] = self['cn']
            kernels['pn'] = staircase_leading_elts(self['pn'])
            kernels['bin_times'] = self['bin_times']
            kernels['kernel_diracs'] = staircase_leading_elts(self['kernel_diracs'])
            kernels['dirac_times'] = self['dirac_times']

            ar.create_group('kernels_part')
            kernels_part = ar['kernels_part']
            kernels_part['kernels'] = staircase_leading_elts(self['kernels_part_save'])
            kernels_part['pn'] = staircase_leading_elts(self['pn_part_save'])
            kernels_part['cn'] = self['cn_part_save']
            kernels_part['bin_times'] = self['bin_times_part']
            kernels_part['size_part'] = self.size_part
            kernels_part['nb_bins_sum'] = self.nb_bins_sum
            kernels_part['kernel_diracs'] = staircase_leading_elts(self['kernel_diracs_part_save'])
            kernels_part['dirac_times'] = self['dirac_times']

            if 'on' in self:
                ar.create_group('green_function')
                green_function = ar['green_function']
                green_function['times'] = params['measure_times']
                green_function['on'] = self['on']
                green_function['on_error'] = self['on_error']
                green_function['sn'] = staircase_leading_elts(self['sn'])
                green_function['sn_error'] = self['sn_error']
                green_function['cn'] = self['cn']
                green_function['cn_error'] = self['cn_error']
                green_function['pn'] = staircase_leading_elts(self['pn'])

        print 'Saved in', filename


class SolverPython(object):

    def __init__(self, g0_lesser, g0_greater, parameters, filename=None):
        self.g0_lesser = g0_lesser
        self.g0_greater = g0_greater
        self.parameters = parameters.copy()
        self.world = MPI.COMM_WORLD
        if 'size_part' not in self.parameters: self.parameters['size_part'] = 10
        if 'nb_bins_sum' not in self.parameters: self.parameters['nb_bins_sum'] = 10
        self.res = Results(self.world, datetime.now(), self.parameters['size_part'],
                           self.parameters['nb_bins_sum'])
        del self.parameters['size_part']
        del self.parameters['nb_bins_sum']

        self.filename = filename

    def order_zero(self):
        params = self.parameters.copy()
        params['U'] = float('NaN')
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        c0, s0 = S.order_zero
        self.res['c0'] = c0

    def checkpoint(self, solver_core, compute=False):
        if self.world.rank == 0:
            print 'Checkpoint order {0} ({1})'.format(solver_core.max_order, datetime.now())
            print 'pn:', solver_core.pn

        self.res.fill_last_slot(solver_core, compute)

        if self.world.rank == 0 and self.filename is not None:
            self.res.save(self.filename, self.parameters)
            print

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
            print '\n* Pre run'
        S.run(nb_warmup_cycles, True)
        prerun_duration = float(clock() - start)
        prerun_duration = self.world.allreduce(prerun_duration) / float(self.world.size) # take average
        if self.world.rank == 0:
            print 'pn:', S.pn
        new_U = calc_ideal_U(S.pn, U)

        # time estimation
        if save_period > 0:
            save_period = max(save_period, 60) # 1 minute mini
            nb_cycles_per_subrun = int(float(nb_warmup_cycles * save_period) / prerun_duration + 0.5)
        else:
            nb_cycles_per_subrun = nb_cycles

        if self.world.rank == 0:
            print 'Nb cycles per subrun =', nb_cycles_per_subrun
            est_run_time = prerun_duration * float(nb_cycles + nb_warmup_cycles) / float(nb_warmup_cycles)
            print 'Estimated run time =', timedelta(seconds=est_run_time)

        # prepare new solver taking prerun into account
        if self.world.rank == 0:
            print 'Changing U:', U, '=>', new_U
        params['U'] = new_U
        S = SolverCore(**params)
        S.set_g0(self.g0_lesser, self.g0_greater)

        # warmup
        if self.world.rank == 0:
            print '\n* Warmup'
        S.run(nb_warmup_cycles, False)

        # main run
        if self.world.rank == 0:
            print '\n* Main runs'
        while S.nb_measures < nb_cycles:
            nb_cycles_remaining = nb_cycles - S.nb_measures
            if nb_cycles_remaining > nb_cycles_per_subrun:
                S.run(nb_cycles_per_subrun, True)
                self.checkpoint(S)
            else: # last run
                S.run(nb_cycles_remaining, True)
                S.compute_sn_from_kernels
                self.checkpoint(S, compute=True)


######################### Front End #######################

def single_solve(g0_lesser, g0_greater, parameters_, filename=None, save_period=-1, histogram=False):
    if histogram:
        raise NotImplementedError

    parameters = parameters_.copy()
    nb_warmup_cycles = parameters.pop('n_warmup_cycles')
    nb_cycles = parameters.pop('n_cycles')

    solver = SolverPython(g0_lesser, g0_greater, parameters, filename)
    solver.order_zero()
    solver.prerun_and_run(nb_warmup_cycles, nb_cycles, parameters['max_perturbation_order'],
                          parameters['U'], save_period=save_period)

    return np.squeeze(solver.res['on']), np.squeeze(solver.res['on_error'])


def staircase_solve(g0_lesser, g0_greater, parameters_, filename=None, save_period=-1):
    parameters = parameters_.copy()
    nb_warmup_cycles = parameters.pop('n_warmup_cycles')
    nb_cycles = parameters.pop('n_cycles')
    parameters["min_perturbation_order"] = 0
    max_order = parameters["max_perturbation_order"]
    U = parameters['U']

    orders = list(range(1, max_order + 1))
    if 'forbid_parity_order' in parameters and parameters['forbid_parity_order'] != -1:
        orders = orders[parameters['forbid_parity_order']::2]

    solver = SolverPython(g0_lesser, g0_greater, parameters, filename)
    solver.order_zero()

    for i, k in enumerate(orders):
        if solver.world.rank == 0:
            print "---------------- Order", k, "----------------"
        solver.prerun_and_run(nb_warmup_cycles, nb_cycles, k, U, save_period=save_period)
        U = solver.res['U'][-1]

    return np.squeeze(solver.res['on']), np.squeeze(solver.res['on_error'])


if __name__ == '__main__':

    # test reduce_binning
    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 5, 1), np.array([[10, 35, 60], [85, 110, 135]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 6, 1), np.array([[15, 51], [105, 141]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    world = MPI.COMM_WORLD
    if world.size != 4:
        raise RuntimeError('These tests must be run with MPI on 4 processes') # for following tests

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
        weight = 100
    elif world.rank == 1:
        array = np.array([[4.5, 1.], [-6.3, 0], [9.4, 1]])
        weight = 75.
    elif world.rank == 2:
        array = np.array([[-86.4, 0.], [40, 0.], [63.1, 0]])
        weight = 0
    elif world.rank == 3:
        array = np.array([[10, 0.], [20, 0], [30, 0]])
        weight = 60

    error_ref = [[1.6809385918497846, 0.2330734287256026], [16.251235980435649, 0.], [5.90993608255, 0.218020229063]]
    error = variance_error(array, world, weight=weight)
    # print error
    assert(error.shape == (3, 2))
    assert(np.allclose(error, error_ref, atol=1e-5))

    # test variance_error
    if world.rank == 0:
        array = np.array([[12.3 +0j, 0.], [68., 0.], [0.52, 1]])
        weight = 0
    elif world.rank == 1:
        array = np.array([[4.5, 1.], [-6.3, 0], [9.4, 1]])
        weight = 0
    elif world.rank == 2:
        array = np.array([[-86.4, 0.], [40, 0.], [63.1, 0]])
        weight = 0
    elif world.rank == 3:
        array = np.array([[10, 0.], [20, 0], [30, 0]])
        weight = 0

    error_ref = [[20.688855695760459, 0.21650635094610965], [13.603693202582892, 0.], [12.033703451140882, 0.25]]
    error = variance_error(array, world, weight=weight)
    # print error
    assert(error.shape == (3, 2))
    assert(np.allclose(error, error_ref, atol=1e-5))

