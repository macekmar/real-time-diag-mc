from mpi4py import MPI
from ctint_keldysh import SolverCore
import numpy as np
from pytriqs.archive import HDFArchive
from datetime import datetime, timedelta
from time import clock
import cPickle
from os.path import splitext
from copy import deepcopy
from scipy.stats import linregress

def expand_axis(a, val, end=False, axis=0):
    """
    Expand the axis `axis` of array `a` with `val`.
    Returns an array with same shape as `a` but increased by one along axis `axis`.
    `val` is used to fill in the thus openned positions.
    """
    # FIXME: is this the same as np.insert(a, 0, val, axis=0) ???
    shape = list(a.shape)
    shape[axis] = 1
    if end:
        return np.append(a, np.tile(val, shape), axis=axis)
    else:
        return np.append(np.tile(val, shape), a, axis=axis)

def squeeze_except(a, except_axes):
    """
    Like np.squeeze but ignoring axes in the list `except_axis`.
    """
    shape = list(a.shape)
    for d in range(a.ndim):
        if d not in except_axes and shape[d] == 1:
            shape[d] = -1
    k = 0
    while k < len(shape):
        if shape[k] == -1:
            del shape[k]
        else:
            k += 1

    return a.reshape(tuple(shape))


def variance_error(data, comm=MPI.COMM_WORLD, weight=None, unbiased=True):
    """
    Compute the error on `data` through sqrt(var(data)/(nb of proccess-1)), where the variance is calculated over
    the processes of communicator `comm`.
    `weight` can be a number (the weight of this process), an array with same shape as `data`, or None (same
    weights for all).
    Returns an array of complex with same shape as `data`.
    """
    if comm.size == 1:
        return np.zeros_like(data)

    weighted = weight is not None
    if comm.rank != 0:
        comm.gather(data)
        if weighted: comm.gather(weight)
        data_error = comm.bcast(None)
    else:
        data_all = np.array(comm.gather(data), dtype=complex)
        if weighted:
            weight_all = np.array(comm.gather(weight))
            weight_sum = np.sum(weight_all, axis=0)

            # do not consider weight which sums to 0
            if (weight_all.ndim == 1):
                if (weight_sum == 0):
                    weight_all = None
                    weighted = False
            else:
                weight_all[:, weight_sum == 0] = 1.
                weight_sum[weight_sum == 0] = float(comm.size)

        else:
            weight_all = None

        avg_data_all = np.average(data_all, weights=weight_all, axis=0)
        var_data_all = np.average(np.abs(data_all - avg_data_all[np.newaxis, ...])**2, weights=weight_all, axis=0)

        ### unbiased estimator
        if unbiased:
            if not weighted:
                factor = comm.size / float(comm.size - 1)
            else:
                factor = weight_sum**2 / float(weight_sum**2 - np.sum(weight_all**2, axis=0))

        data_error = np.sqrt(factor * var_data_all / float(comm.size))

        comm.bcast(data_error)

    return data_error

def save_configuration_list(solver, filename, comm=MPI.COMM_WORLD):
    """
    Pickle stored configuration data from cpp solver as a dictionnary.
    File is named filename_config_rank.pkl
    """
    filename = filename + '_config_' + str(comm.rank).zfill(len(str(comm.size-1))) + '.pkl'

    config_list = solver.config_list
    config_weight = solver.config_weight
    config_mult = solver.config_mult

    config_dict = {'configurations': config_list, 'weights': config_weight, 'multiplicities': config_mult}

    with open(filename, 'wb') as f:
        cPickle.dump(config_dict, f)



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

def calc_ideal_U(pn, U):
    """calculates ideal U value"""
    # FIXME: raises warnings. Make it simpler and cuter.
    if len(pn) < 2:
        raise ValueError('pn must be at least of length 2')
    an = [max(pn[0], pn[1])]
    for k in range(1, len(pn)-1):
        an.append(0.5*(max(pn[k-1], pn[k]) + max(pn[k], pn[k+1])))
    an.append(max(pn[-2], pn[-1]))
    slope, _, _, _, _ = linregress(np.arange(len(an)), np.log(an))
    return 2. * U * np.exp(-slope)

def reduce_binning(x, chunk_size, axis=-1):
    """
    Reduces the size of ND array `x` along dimension `axis` by summing together
    chunks of size `chunk_size`.
    """
    x = np.swapaxes(x, axis, 0)
    shape = (-1, chunk_size) + x.shape[1:]
    x = x[:x.shape[0] - x.shape[0] % chunk_size].reshape(shape).sum(axis=1)
    return np.swapaxes(x, axis, 0)


def _collect_results(solver, res_structure, size_part, comm=MPI.COMM_WORLD):
    """
    Cumulate results of the cpp solver according to a process partitionning.
    Results are returned only by `comm` master proc, in a dictionnary.

    The partitionning is given by `size_part`, the nb of *equal* parts in the partition.
    Details of what and how it is collected is given in `res_structure`.
    The communicator `comm` should be the same as the one with which the cpp
    solver has been run.

    This function can be run several times, but with decreasing partition size
    only, and making sure the old partition is nested in the new one. Indeed
    cumulation of results is not (always) reversible. (#FIXME)
    """
    is_part_master = solver.collect_results(size_part)

    part_comm = comm.Split(is_part_master, comm.rank)
    if not is_part_master:
        part_comm.Free()
    else:
        assert part_comm.size == size_part

    output = {}
    change_axis = lambda x: np.rollaxis(x, 0, x.ndim) # send first axis to last position

    for key in res_structure:
        if isinstance(key, tuple):
            key_coord, key = key

            if comm.rank == 0:
                output[key_coord] = getattr(solver, key_coord)

        if is_part_master:
            data = part_comm.gather(getattr(solver, key), 0)
        if comm.rank == 0:
            output[key] = change_axis(np.array(data))

    if comm.rank == 0:
        return output

def extract_results(solver, results, res_name, res_structure, params, append):
    """
    Extract results from the cpp solver and place them in the list `results`.

    Full results (cumulated over all proc) and partial (over a partition) are
    extracted, as well as some metadata. These are placed in a dictionnary and
    appended to `results`, if `append` is True, or replaces `results[-1]`
    otherwise.  The dictionnary contains three keys: `res_name`,
    `res_name`+'_part' and 'metadata'.  Details of what and how it is collected
    is given in `res_structure`.
    """

    ### extract
    ### the order of these two lines matters !
    res_part = _collect_results(solver, res_structure, params['size_part'])
    res_all = _collect_results(solver, res_structure, 1)

    res = {}
    res[res_name] = res_all
    res[res_name + '_part'] = res_part

    ### metadata that are different in each subrun
    res['metadata'] = {}
    res['metadata']['max_order'] = solver.max_order
    res['metadata']['U'] = solver.U
    res['metadata']['durations'] = solver.qmc_duration
    res['metadata']['nb_measures'] = solver.nb_measures

    ### append or replace
    if append:
        results.append(res)
    else:
        results[-1] = res


def _add_cn_to_results(results, res_name):
    """
    Add cn to `results` last slot.

    cn is calculated using pn and U of the last run, and cn calculated for the
    previous run if it exists. If not, c0 is taken to be 1.

    Recall the formula:
    c_n = c_{n-k} p_{n} / (p_{n-k} U^k)
    where n-k is the largest order of the previous run.

    Note: if runs are separated by more than one non-zero order, it may not be
    the best formula, statistically speaking.
    """
    if len(results) > 1:
        cn = results[-2][res_name]['cn']
        new_pn = results[-1][res_name]['pn'][len(cn)-1:]
        new_cn = cn[-1] * new_pn[1:] / (new_pn[0] * results[-1]['metadata']['U'] ** np.arange(1, len(new_pn)))
        results[-1][res_name]['cn'] = np.append(cn, new_cn)
    else:
        c0 = 1.
        pn = results[-1][res_name]['pn']
        cn = c0 * pn[1:] / (pn[0] * results[-1]['metadata']['U'] ** np.arange(1, len(pn)))
        results[-1][res_name]['cn'] = expand_axis(cn, c0, end=False, axis=0)

def _save_in_archive(results, res_name, res_structure, archive, bin_reduction=False, nb_bins_sum=None):
    """
    Saves a reduced version of the `res_name` data of `results` in `archive`.

    `res_structure` tells what is saved and if/how it is reduced.
    `archive` may be any dictionnary-like object, more specifically an open hdf5 archive.
    If `bin_reduction` is True, time-dependant data is reduced by summing over chunks of size `nb_bins_sum`.

    Order-dependant data are gathered along the different runs (the different slots of `results`).
    """
    for key in res_structure:
        if isinstance(key, tuple):
            key_coord, key_data = key
            coord = results[-1][res_name][key_coord]

            data = results[0][res_name][key_data]
            for res in results[1:]:
                # data.append(res[res_name][key_data][len(data):])
                data = np.concatenate((data, res[res_name][key_data][len(data):]), axis=0)
            # data = np.array(data)

            if bin_reduction:
                coord = reduce_binning(coord, nb_bins_sum) / float(nb_bins_sum)
                data = reduce_binning(data, nb_bins_sum, axis=1) / float(nb_bins_sum)

            archive[key_data] = squeeze_except(data, [0, 1])
            archive[key_coord] = coord

        else:
            data = list(results[0][res_name][key])
            for res in results[1:]:
                data.append(res[res_name][key][len(data):])
            archive[key] = squeeze_except(np.array(data), [0])

def refine_and_save_results(results, res_name, res_structure, params):
    """
    """
    res_structure_tmp = deepcopy(res_structure)

    _add_cn_to_results(results, res_name)
    _add_cn_to_results(results, res_name + '_part')
    res_structure_tmp.append('cn')

    ### create archive and fill it with leading elements and metadata
    with HDFArchive(params['filename'], 'w') as ar:

        ### metadata
        ar.create_group('metadata')
        metadata = ar['metadata']

        for key in ['max_order', 'U', 'durations', 'nb_measures']:
            metadata[key] = np.array([res['metadata'][key] for res in results])

        metadata['nb_proc'] = MPI.COMM_WORLD.size
        metadata['interaction_start'] = params['interaction_start']
        metadata['start_time'] = str(params['start_time'])
        metadata['run_time'] = (datetime.now() - params['start_time']).total_seconds()

        ### data
        ar.create_group(res_name)
        res_ar = ar[res_name]
        ar.create_group(res_name + '_part')
        res_part_ar = ar[res_name + '_part']

        _save_in_archive(results, res_name, res_structure_tmp, res_ar)
        _save_in_archive(results, res_name + '_part', res_structure_tmp, res_part_ar,
                         bin_reduction=True, nb_bins_sum=params['nb_bins_sum'])

        print 'Saved in {0}\n'.format(params['filename'])

###############################################################################

### Parameters and default values. None means there is no default value and the
### parameter is required.
### They are checked for at the beginning of the python code, so the cpp default
### values are overriden (I don;t know how to get them so no choice)
PARAMS_PYTHON_KEYS = {'start_time': None,
                      'staircase': None,
                      'nb_cycles': None,
                      'nb_warmup_cycles': None,
                      'save_period': None,
                      'filename': None,
                      'g0_lesser': None,
                      'g0_greater': None,
                      'size_part': 1,
                      'nb_bins_sum': 1}

### The following params should be the same as in parameters.hpp (and the default values too !)
PARAMS_CPP_KEYS = {'creation_ops': None,
                   'annihilation_ops': None,
                   'extern_alphas': None,
                   'nonfixed_op': False,
                   'interaction_start': None,
                   'alpha': None,
                   'nb_orbitals': None,
                   'U': None,
                   'w_ins_rem': 1.,
                   'w_dbl': 0.5,
                   'w_shift': 0.,
                   'max_perturbation_order': 3,
                   'min_perturbation_order': 0,
                   'forbid_parity_order': -1,
                   'length_cycle': 50,
                   'random_seed': 34788 + 928374 * MPI.COMM_WORLD.rank,
                   'random_name': '',
                   'max_time': -1,
                   'verbosity': 0,
                   'method': 5,
                   'nb_bins': 10000,
                   'singular_thresholds': None,
                   'cycles_trapped_thres': 100,
                   'store_configurations': 0}

def solve(params):
    world = MPI.COMM_WORLD

    ### start time
    params['start_time'] = datetime.now()

    ### manage parameters
    def check_params(params, reference):
        for key, default in reference.items():
            if key not in params:
                if default is None:
                    raise ValueError, "Parameter '{0}' is missing !".format(key)
                else:
                    params[key] = default
                    if world.rank == 0:
                        print "Parameter {0} defaulted to {1}".format(key, str(default))

    params_cpp = params.copy()
    check_params(params, PARAMS_PYTHON_KEYS)
    check_params(params, PARAMS_CPP_KEYS)
    for key in PARAMS_PYTHON_KEYS:
        del params_cpp[key]

    ### manage staircase
    if params['staircase']:
        orders = np.arange(params['min_perturbation_order'] + 1, params['max_perturbation_order'] + 1)
        if params['forbid_parity_order'] != -1:
            orders = orders[orders % 2 == params['forbid_parity_order']]
    else:
        orders = [params['max_perturbation_order']]

    ### result structure
    results = []
    if params['method'] != 0:
        res_name = 'kernels'
        res_structure = ['pn', ('bin_times', 'kernels'), ('bin_times', 'nb_kernels'),
                ('dirac_times', 'kernel_diracs')]
    else:
        res_name = 'green_function'
        res_structure = ['sn', 'pn']


    ### loop over orders
    for k in orders:
        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        params_cpp['max_perturbation_order'] = k
        S = SolverCore(**params_cpp)
        S.set_g0(params['g0_lesser'], params['g0_greater'])

        ### prerun
        start = clock()
        if world.rank == 0:
            print '\n* Pre run'
        S.run(params['nb_warmup_cycles'], True)
        prerun_duration = world.allreduce(float(clock() - start)) / float(world.size) # take average
        if world.rank == 0:
            print 'pn (node 0):', S.pn
        new_U = calc_ideal_U(world.allreduce(S.pn), S.U)

        ### time estimation
        save_period = params['save_period']
        if save_period > 0:
            save_period = max(save_period, 60) # 1 minute mini
            nb_cycles_per_subrun = int(float(params['nb_warmup_cycles'] * save_period) / prerun_duration + 0.5)
        else:
            nb_cycles_per_subrun = params['nb_cycles']

        if world.rank == 0:
            print 'Nb cycles per subrun =', nb_cycles_per_subrun
            est_run_time = prerun_duration * float(params['nb_cycles'] + params['nb_warmup_cycles']) \
                            / float(params['nb_warmup_cycles'])
            print 'Estimated run time =', timedelta(seconds=est_run_time)

        ### prepare new solver taking prerun into account
        if world.rank == 0:
            print 'Changing U:', S.U, '=>', new_U
        params_cpp['U'] = new_U # is also used as a first guess U in next order
        S = SolverCore(**params_cpp)
        S.set_g0(params['g0_lesser'], params['g0_greater'])

        ### warmup
        if world.rank == 0:
            print '\n* Warmup'
        S.run(params['nb_warmup_cycles'], False)

        ### main run
        if world.rank == 0:
            print '\n* Main runs'

        nb_cycles_left = params['nb_cycles']
        append = True
        while nb_cycles_left > 0:
            nb_cycles_todo = min(nb_cycles_per_subrun, nb_cycles_left)
            S.run(nb_cycles_todo, True)
            nb_cycles_left -= nb_cycles_todo

            extract_results(S, results, res_name, res_structure, params, append)
            append = False

            if world.rank == 0 and params['filename'] is not None:
                refine_and_save_results(results, res_name, res_structure, params)

    if params['store_configurations'] > 0 and not params['staircase']:
        save_configuration_list(S, splitext(params['filename'])[0])

    return results




if __name__ == '__main__':
    import warnings

    ### test reduce_binning
    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 5, 1), np.array([[10, 35, 60], [85, 110, 135]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    a = np.arange(30).reshape(2, 15)
    assert np.array_equal(reduce_binning(a, 6, 1), np.array([[15, 51], [105, 141]]))
    assert np.array_equal(a, np.arange(30).reshape(2, 15))

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [0]).shape == (1, 5, 4)

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [1]).shape == (5, 4)

    ### test squeeze_except
    a = np.arange(20).reshape(1, 5, 1, 4)
    assert squeeze_except(a, [1, 2]).shape == (5, 1, 4)

    ### test squeeze_except
    a = np.array([]).reshape(0, 5, 1, 4)
    assert a.shape == (0, 5, 1, 4)
    assert squeeze_except(a, [1]).shape == (0, 5, 4)

    world = MPI.COMM_WORLD
    if world.size == 4:

        ### test variance_error
        if world.rank == 0:
            array = np.array([12.3, 68., 0.52])
        elif world.rank == 1:
            array = np.array([4.5, -6.3, 9.4])
        elif world.rank == 2:
            array = np.array([-86.4, 40, 63.1])
        elif world.rank == 3:
            array = np.array([10, 20, 30])

        error_ref = np.array([20.688855695760459, 13.603693202582892, 12.033703451140882]) * np.sqrt(4/3.)
        error = variance_error(array, world)
        # print error_ref
        # print error
        assert(error.shape == (3,))
        assert(np.allclose(error, error_ref, atol=1e-5))

        ### test variance_error
        if world.rank == 0:
            array = np.array([[12.3, 0.], [68., 0.], [0.52, 1]])
        elif world.rank == 1:
            array = np.array([[4.5, 1.], [-6.3, 0], [9.4, 1]])
        elif world.rank == 2:
            array = np.array([[-86.4, 0.], [40, 0.], [63.1, 0]])
        elif world.rank == 3:
            array = np.array([[10, 0.], [20, 0], [30, 0]])

        error_ref = np.array([[20.688855695760459, 0.21650635094610965], [13.603693202582892, 0.], [12.033703451140882, 0.25]]) * np.sqrt(4/3.)
        error = variance_error(array, world)
        assert(error.shape == (3, 2))
        assert(np.allclose(error, error_ref, atol=1e-5))

        ### test variance_error
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

        error_ref = np.array([[1.6809385918497846, 0.2330734287256026], [16.251235980435649, 0.], [5.90993608255, 0.218020229063]]) * np.sqrt(235.**2 / float(235.**2 - (100**2 + 75**2 + 60**2)))
        error = variance_error(array, world, weight=weight)
        # print error
        assert(error.shape == (3, 2))
        assert(np.allclose(error, error_ref, atol=1e-5))

        ### test variance_error
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

        error_ref = np.array([[20.688855695760459, 0.21650635094610965], [13.603693202582892, 0.], [12.033703451140882, 0.25]]) * np.sqrt(4/3.)
        error = variance_error(array, world, weight=weight)
        # print error
        assert(error.shape == (3, 2))
        assert(np.allclose(error, error_ref, atol=1e-5))

    else:
        warnings.warn("Tests of 'variance_error' must be run with MPI on 4 processes. Some tests have not been run.", RuntimeWarning)

    ### test _add_cn_to_results
    results = [{'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}]
    _add_cn_to_results(results, 'Paul')
    assert len(results) == 1
    assert np.array_equal(results[0]['Paul']['pn'], np.array([1, 5, 18, 39]))
    assert results[0]['metadata']['U'] == 0.8
    assert results[0]['Paul']['cn'].shape == (4,)
    # print results[0]['Paul']['cn']
    assert np.allclose(results[0]['Paul']['cn'], np.array([1., 6.25, 28.125, 76.171875]))

    ### test _add_cn_to_results
    ### should work with extra axes
    results = [{'Paul': {'pn': np.array([[1], [5]])}, 'metadata': {'U': 0.8}}]
    _add_cn_to_results(results, 'Paul')
    assert len(results) == 1
    assert np.array_equal(results[0]['Paul']['pn'], np.array([[1], [5]]))
    assert results[0]['metadata']['U'] == 0.8
    assert results[0]['Paul']['cn'].shape == (2, 1)
    # print results[0]['Paul']['cn']
    assert np.allclose(results[0]['Paul']['cn'], np.array([[1.], [6.25]]))

    ### test _add_cn_to_results
    run1 = {'Paul': {'pn': np.array([10, 32]), 'cn': np.array([1., 6.4])}, 'metadata': {'U': 0.5}}
    run2 = {'Paul': {'pn': np.array([4, 27, 56]), 'cn': np.array([1., 6.4, 22.123456790123456])}, 'metadata': {'U': 0.6}}
    run3 = {'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}
    results = [run1, run2, run3]
    _add_cn_to_results(results, 'Paul')
    assert len(results) == 3
    assert np.array_equal(results[-1]['Paul']['pn'], np.array([1, 5, 18, 39]))
    assert results[-1]['metadata']['U'] == 0.8
    assert results[-1]['Paul']['cn'].shape == (4,)
    assert np.allclose(results[-1]['Paul']['cn'], np.array([1., 6.4, 22.123456790123456, 59.917695473251015]))

    ### test _add_cn_to_results
    run1 = {'Paul': {'pn': np.array([10, 0, 32]), 'cn': np.array([1., 0, 12.5])}, 'metadata': {'U': 0.5}}
    run2 = {'Paul': {'pn': np.array([4, 0, 27, 0, 56]), 'cn': np.array([1., 0, 12.5, 0, 73.74485596707818])}, 'metadata': {'U': 0.6}}
    run3 = {'Paul': {'pn': np.array([1, 0, 5, 0, 18, 0, 39])}, 'metadata': {'U': 0.8}}
    results = [run1, run2, run3]
    _add_cn_to_results(results, 'Paul')
    assert len(results) == 3
    assert np.array_equal(results[-1]['Paul']['pn'], np.array([1, 0, 5, 0, 18, 0, 39]))
    assert results[-1]['metadata']['U'] == 0.8
    assert results[-1]['Paul']['cn'].shape == (7,)
    assert np.allclose(results[-1]['Paul']['cn'], np.array([1., 0, 12.5, 0, 73.74485596707818, 0, 249.65706447187918]))

    ### test _save_in_archive
    pn = np.array([[3], [12]])
    t = np.linspace(0, 10, 5)
    vn = np.linspace(0, 10, 5)*1.j
    vn = np.vstack((vn, 2*vn)) # of shape (2, 5)
    vn = vn.reshape((2, 5, 1))
    run1 = {'Paul': {'pn': pn, 't': t, 'vn': vn}, 'Jacques': {'bla': None}}
    pn = np.array([[5], [31], [99]])
    vn = np.linspace(1, 4, 5)*1.j
    vn = np.vstack((vn, 2*vn, 3*vn)) # of shape (3, 5)
    vn = vn.reshape((3, 5, 1))
    run2 = {'Paul': {'pn': pn, 't': t, 'vn': vn}, 'Jacques': {'bla': None}}
    results = [run1, run2]
    res_structure = ['pn', ('t', 'vn')]

    archive = {} # simulate an archive with a dict
    _save_in_archive(results, 'Paul', res_structure, archive)

    pn = np.array([3, 12, 99])
    t = np.linspace(0, 10, 5)
    v1 = np.linspace(0, 10, 5)*1.j
    v2 = 2*v1
    v3 = 3*np.linspace(1, 4, 5)*1.j
    vn = np.vstack((v1, v2, v3)) # of shape (3, 5)
    archive_ref = {'pn': pn, 't': t, 'vn': vn}
    assert len(archive) == 3
    assert np.array_equal(archive['pn'], archive_ref['pn'])
    assert np.array_equal(archive['t'], archive_ref['t'])
    assert np.array_equal(archive['vn'], archive_ref['vn'])

