from mpi4py import MPI
from ctint_keldysh import SolverCore
import numpy as np
from pytriqs.archive import HDFArchive
from datetime import datetime, timedelta
from time import clock
import cPickle
from os.path import splitext
from scipy.stats import linregress
from utility import squeeze_except, reduce_binning
import warnings

### Complex numbers are everywhere in this solver, one should not mess with them
warnings.filterwarnings('error', '', np.ComplexWarning)

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
        assert part_comm.size == min(size_part, comm.size)

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

def extract_results(solver, results, res_structure, params, append):
    """
    Extract results from the cpp solver and place them in the list `results`.

    Full results (cumulated over all proc) and partial (over a partition) are
    extracted, as well as some metadata. These are placed in a dictionnary and
    appended to `results`, if `append` is True, or replaces `results[-1]`
    otherwise.  The dictionnary contains three keys: 'results_all',
    'results_part' and 'metadata'.  Details of what and how it is collected
    is given in `res_structure`.
    """

    ### extract
    ### the order of these two lines matters !
    res_part = _collect_results(solver, res_structure, params['size_part'])
    res_all = _collect_results(solver, res_structure, 1)

    res = {}
    res['results_all'] = res_all
    res['results_part'] = res_part

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


def add_cn_to_results(results, res_name):
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
    U = results[-1]['metadata']['U']
    pn = results[-1][res_name]['pn']

    if len(results) > 1:
        prev_cn = results[-2][res_name]['cn']
        cn = np.zeros_like(pn, dtype=float)[:len(pn)-len(prev_cn)]
        cn = np.concatenate((prev_cn, cn), axis=0)
        for k in range(len(prev_cn), len(cn)):
            i = 1
            while (pn[k-i] == 0.).any() and i < k:
                i += 1
            cn[k] = cn[k-i] * pn[k] / (pn[k-i] * U**i)

    else:
        c0 = 1.
        cn = np.zeros_like(pn, dtype=float)
        cn[0] = c0
        for k in range(1, len(pn)):
            i = 1
            while (pn[k-i] == 0).any() and i < k:
                i += 1
            cn[k] = cn[k-i] * pn[k] / (pn[k-i] * U**i)

    results[-1][res_name]['cn'] = cn


def _save_in_archive(results, res_name, archive, bin_reduction=False, nb_bins_sum=None):
    """
    Saves a reduced version of the `res_name` data of `results` in `archive`.

    `archive` may be any dictionnary-like object, more specifically an open hdf5 archive.
    If `bin_reduction` is True, time-dependant data is reduced by summing over chunks of size `nb_bins_sum`.
    Order-dependant data are gathered along the different runs (the different slots of `results`).
    """

    for key in results[-1][res_name]:
        if key in ['pn', 'cn', 'sn']:
            data = results[0][res_name][key]
            for res in results[1:]:
                data = np.concatenate((data, res[res_name][key][len(data):]), axis=0)
            archive[key] = squeeze_except(data, [0])

        elif key == 'dirac_times':
            archive[key] = results[-1][res_name][key]

        elif key == 'bin_times':
            data = results[-1][res_name][key]
            if bin_reduction:
                data = reduce_binning(data, nb_bins_sum) / float(nb_bins_sum)
            archive[key] = data

        elif key == 'kernels':
            data = results[0][res_name][key]
            for res in results[1:]:
                data = np.concatenate((data, res[res_name][key][len(data):]), axis=0)

            if bin_reduction:
                data = reduce_binning(data, nb_bins_sum, axis=1) / float(nb_bins_sum)

            archive[key] = squeeze_except(data, [0, 1])

        elif key == 'nb_kernels':
            data = results[0][res_name][key]
            for res in results[1:]:
                data = np.concatenate((data, res[res_name][key][len(data):]), axis=0)

            if bin_reduction:
                data = reduce_binning(data, nb_bins_sum, axis=1) # no normalization !

            archive[key] = squeeze_except(data, [0, 1])

        elif key == 'kernel_diracs':
            data = results[0][res_name][key]
            for res in results[1:]:
                data = np.concatenate((data, res[res_name][key][len(data):]), axis=0)

            archive[key] = squeeze_except(data, [0, 1])

        else:
            warnings.warn('An unknown result key has been found. It has not been saved.', RuntimeWarning)

def _next_name_gen(first_name):
    """
    Generates sequence of strings like, if `first_name` is "blabla":
    blabla_1
    blabla_2
    blabla_3
    ...

    If `first_name` already has a trailing number seprarated by an underscore,
    it is increased. Leading zeros are kept if present.
    """
    import re
    regex = re.compile(r'_[0-9]+$')
    name = first_name
    while True:
        match = regex.search(name)
        if match is None:
            name = name + '_1'
        else:
            match = match.group()[1:]
            l = len(match)
            n = int(match)
            name = name[:-l] + str(n+1).zfill(l)
        yield name

def refine_and_save_results(results, params, start_time, overwrite=True):
    """
    Saves a refined version of `results` in an HDF5 archive named
    `params['filename']`. Also saves `params` and some metadata.

    Data is saved in a group named `params['run_name']` created in the archive
    or overwritten. The rest of the archive is left untouched. If `overwrite`
    is False, a new name is found so that nothing is overwritten in the
    archive.

    In 'run_name' are created four subgroups:
     > 'results_all': store results gathered over all processes
     > 'results_part': store results gathered over a partition of processes
     (used for error estimation)
     > 'metadata': some data about how the run went
     > 'parameters': a copy of `params`

    """

    ### create archive and fill it with leading elements and metadata
    with HDFArchive(params['filename'], 'a') as ar:

        ### run group
        run_name = params['run_name']
        if run_name in ar:
            if overwrite:
                del ar[run_name]
            else:
                ### Change run_name by appending a (or increasing the) trailling number
                nng = _next_name_gen(run_name)
                run_name = nng(run_name)
                while run_name in ar:
                    run_name = nng(run_name)

        ar.create_group(run_name)
        run = ar[run_name]

        ### metadata
        run.create_group('metadata')
        metadata = run['metadata']

        for key in ['max_order', 'U', 'durations', 'nb_measures']:
            metadata[key] = np.array([res['metadata'][key] for res in results])

        metadata['nb_proc'] = MPI.COMM_WORLD.size
        metadata['interaction_start'] = params['interaction_start']
        metadata['start_time'] = str(start_time)
        metadata['run_time'] = str(datetime.now() - start_time)

        ### parameters
        run.create_group('parameters')
        run['parameters'].update(params)

        ### data
        run.create_group('results_all')
        res_all = run['results_all']
        run.create_group('results_part')
        res_part = run['results_part']

        _save_in_archive(results, 'results_all', res_all)
        _save_in_archive(results, 'results_part', res_part,
                         bin_reduction=True, nb_bins_sum=params['nb_bins_sum'])

    print 'Saved in {0}\n'.format(params['filename'])

###############################################################################

### Parameters and default values. None means there is no default value and the
### parameter is required.
### They are checked for at the beginning of the python code, so the cpp default
### values are overriden (I don't know how to get them so no choice)
PARAMS_PYTHON_KEYS = {'staircase': None,
                      'nb_cycles': None,
                      'nb_warmup_cycles': None,
                      'save_period': None,
                      'filename': None,
                      'run_name': 'run_1',
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
                   'potential': None,
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
                   'method': 1,
                   'nb_bins': 10000,
                   'singular_thresholds': None,
                   'cycles_trapped_thresh': 100,
                   'store_configurations': 0}

def solve(params):
    world = MPI.COMM_WORLD

    ### start time
    start_time = datetime.now()

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

    check_params(params, PARAMS_PYTHON_KEYS)
    params_cpp = params.copy()
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
        res_structure = ['pn', ('bin_times', 'kernels'), ('bin_times', 'nb_kernels'),
                ('dirac_times', 'kernel_diracs')]
    else:
        res_structure = ['sn', 'pn']


    ### loop over orders
    for k in orders:
        if world.rank == 0:
            print "---------------- Order", k, "----------------"

        if k == 1: # at order 1, force disable double moves
            params_cpp['w_dbl'] = 0.
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
            nb_cycles_per_subrun = int(float(params['nb_warmup_cycles'] * save_period) / prerun_duration + 0.5)
            nb_cycles_per_subrun = max(1, nb_cycles_per_subrun) # no empty subrun
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

            extract_results(S, results, res_structure, params, append)
            append = False

            if world.rank == 0:
                print 'pn (all nodes):', S.pn # results have been gathered previously
                add_cn_to_results(results, 'results_all')
                add_cn_to_results(results, 'results_part')
                if params['filename'] is not None:
                    refine_and_save_results(results, params, start_time)

    if params['store_configurations'] > 0 and not params['staircase']:
        save_configuration_list(S, splitext(params['filename'])[0])

    return results




if __name__ == '__main__':
    import warnings
    print 'Start tests'

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

    ### test add_cn_to_results
    results = [{'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 1
    assert np.array_equal(results[0]['Paul']['pn'], np.array([1, 5, 18, 39]))
    assert results[0]['metadata']['U'] == 0.8
    assert results[0]['Paul']['cn'].shape == (4,)
    # print results[0]['Paul']['cn']
    assert np.allclose(results[0]['Paul']['cn'], np.array([1., 6.25, 28.125, 76.171875]))

    ### test add_cn_to_results
    ### should work with extra axes
    results = [{'Paul': {'pn': np.array([[1, 3], [5, 6]])}, 'metadata': {'U': 0.8}}]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 1
    assert np.array_equal(results[0]['Paul']['pn'], np.array([[1, 3], [5, 6]]))
    assert results[0]['metadata']['U'] == 0.8
    assert results[0]['Paul']['cn'].shape == (2, 2)
    # print results[0]['Paul']['cn']
    assert np.allclose(results[0]['Paul']['cn'], np.array([[1., 1.], [6.25, 2.5]]))

    ### test add_cn_to_results
    ### should work with extra axes
    results = [{'Paul': {'pn': np.array([[1, 3], [5, 6], [10, 8]])}, 'metadata': {'U': 0.8}}]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 1
    assert np.array_equal(results[0]['Paul']['pn'], np.array([[1, 3], [5, 6], [10, 8]]))
    assert results[0]['metadata']['U'] == 0.8
    assert results[0]['Paul']['cn'].shape == (3, 2)
    # print results[0]['Paul']['cn']
    assert np.allclose(results[0]['Paul']['cn'], np.array([[1., 1.], [6.25, 2.5], [15.625, 4.166666666666667]]))

    ### test add_cn_to_results
    run1 = {'Paul': {'pn': np.array([10, 32]), 'cn': np.array([1., 6.4])}, 'metadata': {'U': 0.5}}
    run2 = {'Paul': {'pn': np.array([4, 27, 56]), 'cn': np.array([1., 6.4, 22.123456790123456])}, 'metadata': {'U': 0.6}}
    run3 = {'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}
    results = [run1, run2, run3]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 3
    assert np.array_equal(results[-1]['Paul']['pn'], np.array([1, 5, 18, 39]))
    assert results[-1]['metadata']['U'] == 0.8
    assert results[-1]['Paul']['cn'].shape == (4,)
    assert np.allclose(results[-1]['Paul']['cn'], np.array([1., 6.4, 22.123456790123456, 59.917695473251015]))

    ### test add_cn_to_results
    run1 = {'Paul': {'pn': np.array([[10, 12], [32, 28]]),
                     'cn': np.array([[1., 1.], [6.4, 4.666666666666667]])},
            'metadata': {'U': 0.5}}
    run2 = {'Paul': {'pn': np.array([[4, 3], [27, 18], [56, 59]]),
                     'cn': np.array([[1., 1.], [6.4, 4.666666666666667], [22.123456790123456, 25.49382716049383]])},
            'metadata': {'U': 0.6}}
    run3 = {'Paul': {'pn': np.array([[1, 0], [5, 7], [18, 19], [39, 45]])},
            'metadata': {'U': 0.8}}
    results = [run1, run2, run3]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 3
    assert np.array_equal(results[-1]['Paul']['pn'], np.array([[1, 0], [5, 7], [18, 19], [39, 45]]))
    assert results[-1]['metadata']['U'] == 0.8
    # print results[-1]['Paul']['cn']
    assert results[-1]['Paul']['cn'].shape == (4, 2)
    assert np.allclose(results[-1]['Paul']['cn'], np.array([[1., 1.], [6.4, 4.666666666666667], [22.123456790123456, 25.49382716049383], [59.917695473251015, 75.47514619883042]]))

    ### test add_cn_to_results
    run1 = {'Paul': {'pn': np.array([10, 0, 32]), 'cn': np.array([1., 0, 12.5])}, 'metadata': {'U': 0.5}}
    run2 = {'Paul': {'pn': np.array([4, 0, 27, 0, 56]), 'cn': np.array([1., 0, 12.5, 0, 73.74485596707818])}, 'metadata': {'U': 0.6}}
    run3 = {'Paul': {'pn': np.array([1, 0, 5, 0, 18, 0, 39])}, 'metadata': {'U': 0.8}}
    results = [run1, run2, run3]
    add_cn_to_results(results, 'Paul')
    assert len(results) == 3
    assert np.array_equal(results[-1]['Paul']['pn'], np.array([1, 0, 5, 0, 18, 0, 39]))
    assert results[-1]['metadata']['U'] == 0.8
    assert results[-1]['Paul']['cn'].shape == (7,)
    # print results[-1]['Paul']['cn']
    assert np.allclose(results[-1]['Paul']['cn'], np.array([1., 0, 12.5, 0, 73.74485596707818, 0, 249.65706447187918]))

    ### test _save_in_archive
    pn = np.array([[3], [12]])
    bin_times = np.linspace(0, 10, 5)
    kernels = np.linspace(0, 10, 5)*1.j
    kernels = np.vstack((kernels, 2*kernels)) # of shape (2, 5)
    kernels = kernels.reshape((2, 5, 1))
    run1 = {'Paul': {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}, 'Jacques': {'bla': None}}
    pn = np.array([[5], [31], [99]])
    kernels = np.linspace(1, 4, 5)*1.j
    kernels = np.vstack((kernels, 2*kernels, 3*kernels)) # of shape (3, 5)
    kernels = kernels.reshape((3, 5, 1))
    run2 = {'Paul': {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}, 'Jacques': {'bla': None}}
    results = [run1, run2]

    archive = {} # simulate an archive with a dict
    _save_in_archive(results, 'Paul', archive)

    pn = np.array([3, 12, 99])
    bin_times = np.linspace(0, 10, 5)
    v1 = np.linspace(0, 10, 5)*1.j
    v2 = 2*v1
    v3 = 3*np.linspace(1, 4, 5)*1.j
    kernels = np.vstack((v1, v2, v3)) # of shape (3, 5)
    archive_ref = {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}
    assert len(archive) == 3
    assert np.array_equal(archive['pn'], archive_ref['pn'])
    assert np.array_equal(archive['bin_times'], archive_ref['bin_times'])
    assert np.array_equal(archive['kernels'], archive_ref['kernels'])


    ### test _save_in_archive
    ### with 2 parts
    pn = np.array([[3, 4], [12, 17]])
    bin_times = np.linspace(0, 10, 5)
    kernels = np.array([[8.7j, 6.2], [7.4, 4.9j], [9.3, 3.9], [1.2j, 1.5j], [1.0, 1.2]]) # of shape (5, 2)
    kernels = np.array([kernels, 2*kernels]) # of shape (2, 5, 2)
    run1 = {'Paul': {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}, 'Jacques': {'bla': None}}
    pn = np.array([[5, 3], [31, 40], [99, 89]])
    kernels = np.array([[7.5j, 6.1], [6.7, 1.4], [6.4j, 2.1], [9.3j, 0.7j], [6.7, 0.3]]) # of shape (5, 2)
    kernels = np.array([kernels, 2*kernels, 3*kernels]) # of shape (3, 5, 2)
    run2 = {'Paul': {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}, 'Jacques': {'bla': None}}
    results = [run1, run2]

    archive = {} # simulate an archive with a dict
    _save_in_archive(results, 'Paul', archive)

    pn = np.array([[3, 4], [12, 17], [99, 89]])
    bin_times = np.linspace(0, 10, 5)
    v1 = np.array([[8.7j, 6.2], [7.4, 4.9j], [9.3, 3.9], [1.2j, 1.5j], [1.0, 1.2]])
    v2 = 2*v1
    v3 = 3*np.array([[7.5j, 6.1], [6.7, 1.4], [6.4j, 2.1], [9.3j, 0.7j], [6.7, 0.3]])
    kernels = np.array([v1, v2, v3]) # of shape (3, 5, 2)
    archive_ref = {'pn': pn, 'bin_times': bin_times, 'kernels': kernels}
    assert len(archive) == 3
    assert np.array_equal(archive['pn'], archive_ref['pn'])
    assert np.array_equal(archive['bin_times'], archive_ref['bin_times'])
    assert np.array_equal(archive['kernels'], archive_ref['kernels'])

    ### test _next_name_gen
    nng = _next_name_gen('blabla_36')
    assert nng.next() == 'blabla_37'
    assert nng.next() == 'blabla_38'
    assert nng.next() == 'blabla_39'

    nng = _next_name_gen('blabla_036')
    assert nng.next() == 'blabla_037'
    assert nng.next() == 'blabla_038'
    assert nng.next() == 'blabla_039'

    nng = _next_name_gen('blabla_98')
    assert nng.next() == 'blabla_99'
    assert nng.next() == 'blabla_100'
    assert nng.next() == 'blabla_101'

    nng = _next_name_gen('blabla2')
    assert nng.next() == 'blabla2_1'
    assert nng.next() == 'blabla2_2'

    nng = _next_name_gen('blabla')
    assert nng.next() == 'blabla_1'
    assert nng.next() == 'blabla_2'

    print 'Success'
