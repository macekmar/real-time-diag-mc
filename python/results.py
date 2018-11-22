import numpy as np
from utility import pad_along_axis

def _safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))


def merge_results(results1, results2):
    """
    Merge results into a single result dictionnary.

    Inputs are dictionnary-like objects containing three keys:
    > results_all
    > results_part
    > metadata
    They are left unmodified, a new dictionnary is returned.

    If one of the inputs is empty, the other is returned.

    The merging strategy is the following:
    TODO: WIP
    """

    ### empty case
    if len(results1) == 0:
        return results2
    if len(results2) == 0:
        return results1

    ### make sure results2 is going to higher or equal order than results1
    if results1['results_all']['pn'].shape[1] > results2['results_all']['pn'].shape[1]:
        results1, results2 = results2, results1

    extra_orders = results2['results_all']['pn'].shape[1] - results1['results_all']['pn'].shape[1]

    def _merge(res1, res2):
        output = {}

        U1 = pad_along_axis(res1['U'], 0, extra_orders, axis=1,
                            mode='constant', constant_values=0.0)
        U2 = res2['U']
        output['U'] = np.concatenate((U1, U2), axis=0)

        pn1 = pad_along_axis(res1['pn'], 0, extra_orders, axis=1,
                             mode='constant', constant_values=0)
        pn2 = res2['pn']
        output['pn'] = np.concatenate((pn1, pn2), axis=0)
        pn_sum1 = pn1.sum(axis=0)
        pn_sum2 = pn2.sum(axis=0)
        pn_sum = pn_sum1 + pn_sum2

        ### sign
        if 'sn' in res1 and 'sn' in res2:
            sn1 = pad_along_axis(res1['sn'], 0, extra_orders, axis=0,
                                 mode='constant', constant_values=0.0)
            sn2 = res2['sn']
            output['sn'] = _safe_divide(pn_sum1 * sn1 + pn_sum2 * sn2, pn_sum)

        elif 'kernels' in res1 and 'kernels' in res2:

            ### adapt pn shape
            while pn_sum.ndim < res1['kernels'].ndim:
                pn_sum = pn_sum[:, np.newaxis]
                pn_sum1 = pn_sum1[:, np.newaxis]
                pn_sum2 = pn_sum2[:, np.newaxis]

            assert (res1['bin_times'] == res2['bin_times']).all()
            output['bin_times'] = res1['bin_times'].copy()

            data1 = pad_along_axis(res1['kernels'], 0, extra_orders, axis=0,
                                   mode='constant', constant_values=0.0)
            data2 = res2['kernels']
            output['kernels'] = _safe_divide(pn_sum1[1:] * data1 + pn_sum2[1:] * data2, pn_sum[1:])

            data1 = pad_along_axis(res1['nb_kernels'], 0, extra_orders, axis=0,
                                   mode='constant', constant_values=0)
            data2 = res2['nb_kernels']
            output['nb_kernels'] = data1 + data2

            assert (res1['dirac_times'] == res2['dirac_times']).all()
            output['dirac_times'] = res1['dirac_times'].copy()

            data1 = pad_along_axis(res1['kernel_diracs'], 0, extra_orders, axis=0,
                                   mode='constant', constant_values=0.0)
            data2 = res2['kernel_diracs']
            output['kernel_diracs'] = _safe_divide(pn_sum1[1:] * data1 + pn_sum2[1:] * data2, pn_sum[1:])

        else:
            raise ValueError('results have different data types')

        return output

    output = {}
    output['results_all'] = _merge(results1['results_all'], results2['results_all'])
    output['results_part'] = _merge(results1['results_part'], results2['results_part'])

    ### merge metadata
    output['metadata'] = {}

    dur1 = np.pad(results1['metadata']['durations'], (0, extra_orders),
                  mode='constant', constant_values=0.0)
    dur2 = results2['metadata']['durations']
    output['metadata']['durations'] = dur1 + dur2
    output['metadata']['duration'] = results1['metadata']['duration']\
                                    + results2['metadata']['duration']

    output['metadata']['nb_measures'] = results1['metadata']['nb_measures'] \
                                        + results2['metadata']['nb_measures']

    if results1['metadata']['nb_proc'] == results2['metadata']['nb_proc']:
        output['metadata']['nb_proc'] = results1['metadata']['nb_proc']

    return output

def _cn_formula(p_list, U_list, ind_low, ind_high):
    assert p_list[ind_low] != 0
    return p_list[ind_high] / (p_list[ind_low] * np.prod(U_list[ind_low:ind_high]))

def _compute_cn_single_run(pn, Un, c0=1.):
    """
    computes cn from pn and Un, using the best run for each cn.
    `pn` is a (nb_runs, nb_orders) array.
    `Un` is a (nb_runs, nb_orders-1) array.
    returns a 1D array of size `nb_orders`.
    """
    N = pn.shape[1] # nb orders

    cn = np.empty(N, dtype=float)
    cn[0] = c0

    for n in range(1, N):
        prod_pn = pn[:, n][:, np.newaxis] * pn[:, :n]
        if (prod_pn == 0).all():
            cn[n] = 0.
        else:
            run_ind, order_ind = np.unravel_index(np.argmax(prod_pn), prod_pn.shape)
            cn[n] = cn[order_ind] * _cn_formula(pn[run_ind, :], Un[run_ind, :], order_ind, n)

    return cn


def _compute_cn_cum(pn, Un, c0=1.):
    """
    computes cn from pn and Un of several runs.
    `pn` is a (nb_runs, nb_orders) array.
    `Un` is a (nb_runs, nb_orders-1) array.
    returns a 1D array of size `nb_orders`.

    c_n = c_{k} \frac{\sum_i p_{i,n}}{\sum_i p_{i,k} U_{i,k} \ldots U_{i,n-1}}
    for the k<n where \sum_i p_{i,k} is higher (most accurate data).
    """
    M = len(pn) # nb runs
    N = pn.shape[1] # nb orders

    nb_measures = np.sum(pn, axis=0)

    cn = np.empty(N, dtype=float)
    cn[0] = c0

    denom = np.empty(M, dtype=float)
    for n in range(1, N):
        k = np.argmax(nb_measures[:n])

        denom[:] = pn[:, k]
        for kp in range(k, n):
            denom[:] *= Un[:, kp]

        cn[n] = cn[k] * nb_measures[n] / denom.sum()

    return cn

def _compute_cn(pn, Un, c0=1., method='single'):
    if method == 'single':
        return _compute_cn_single_run(pn, Un, c0=c0)
    elif method == 'cum':
        return _compute_cn_cum(pn, Un, c0=c0)

def _compute_cn_v(pn, Un, c0=1., method='single'):
    nb_part = pn.shape[2]

    cn = np.empty(pn.shape[1:], dtype=float)
    for i in range(nb_part):
        cn[:, i] = _compute_cn(pn[:, :, i], Un, c0=c0, method=method)

    return cn


def add_cn_to_results(results, method='single'):
    """
    Computes cn from data in `results` and add it as a new key under
    'results_all' and 'results_part'.
    """

    res = results['results_all']
    res['cn'] = _compute_cn(res['pn'], res['U'], method=method)
    res = results['results_part']
    res['cn'] = _compute_cn_v(res['pn'], res['U'], method=method)

def merge(results1, results2):
    """
    Merge results into a single result dictionnary, including parameters (if it
    makes sense), and compute cn.
    """

    output = merge_results(results1, results2)
    add_cn_to_results(output)

    ### merge parameters if they are equal in both results
    if 'parameters' in results1 and 'parameters' in results2:
        output['parameters'] = {}
        for key, val in results1['parameters'].items():
            if key in results2['parameters']:
                if results2['parameters'][key] == val:
                    output['parameters'][key] = val
                else:
                    output['parameters'][key] = 'undefined'

    return output

if __name__ == '__main__':

    ### test merge_results
    results1 = {'results_all': {}, 'results_part': {}, 'metadata': {}}
    results1['results_all']['pn'] = np.array([[12, 45, 0],
                                              [6, 22, 67]])
    results1['results_all']['U'] = np.array([[0.5, 0.],
                                             [0.8, 0.8]])
    results1['results_all']['sn'] = np.array([-0.5, 0.71, -0.82])
    results1['results_part']['pn'] = np.array([[[5, 7], [22, 23], [0, 0]],
                                               [[2, 4], [9, 13], [30, 37]]])
    results1['results_part']['U'] = np.array([[0.5, 0.],
                                              [0.8, 0.8]])
    results1['results_part']['sn'] = np.array([[-0.45, -0.51], [0.65, 0.72], [-0.82, -0.84]])
    results1['metadata']['nb_measures'] = 152
    results1['metadata']['nb_proc'] = 2
    results1['metadata']['duration'] = 7835
    results1['metadata']['durations'] = np.array([2746, 5089])

    results2 = {'results_all': {}, 'results_part': {}, 'metadata': {}}
    results2['results_all']['pn'] = np.array([[3, 17, 38, 56]])
    results2['results_all']['U'] = np.array([[0.9, 0.9, 0.9]])
    results2['results_all']['sn'] = np.array([-0.3, 0.73, -0.825, 1.2])
    results2['results_part']['pn'] = np.array([[[1, 2], [9, 8], [18, 20], [22, 34]]])
    results2['results_part']['U'] = np.array([[0.9, 0.9, 0.9]])
    results2['results_part']['sn'] = np.array([[-0.32, -0.25], [0.64, 0.80], [-0.84, -0.795], [1.15, 1.23]])
    results2['metadata']['nb_measures'] = 114
    results2['metadata']['nb_proc'] = 2
    results2['metadata']['duration'] = 12837
    results2['metadata']['durations'] = np.array([827, 3839, 8171])

    results_ref= {'results_all': {}, 'results_part': [], 'metadata': {}}
    results_ref['results_all']['pn'] = np.array([[12, 45, 0, 0],
                                                 [6, 22, 67, 0],
                                                 [3, 17, 38, 56]])
    results_ref['results_all']['U'] = np.array([[0.5, 0., 0.],
                                                [0.8, 0.8, 0.],
                                                [0.9, 0.9, 0.9]])
    results_ref['results_all']['sn'] = np.array([-0.4714285714285714, 0.714047619047619, np.nan, np.nan])
    # results_ref['results_part']['pn'] = np.array([[[1, 2], [9, 8], [18, 20], [22, 34]]])
    # results_ref['results_part']['U'] = np.array([[0.9, 0.9, 0.9]])
    # results_ref['results_part']['sn'] = np.array([[-0.32, -0.25], [0.64, 0.80], [-0.84, -0.795], [1.15, 1.23]])
    results_ref['metadata']['nb_measures'] = 266
    results_ref['metadata']['nb_proc'] = 2
    results_ref['metadata']['duration'] = 20672
    results_ref['metadata']['durations'] = np.array([3573, 8928, 8171])
    ### TODO: WIP


    ### test _compute_cn
    pn = np.array([[1, 5, 18, 39]]) # shape = (1, 4)
    Un = np.array([[0.8, 0.8, 0.8]]) # shape = (1, 3)
    cn_ref = np.array([1., 6.25, 28.125, 76.171875])
    cn = _compute_cn(pn, Un, method='single')
    assert cn.shape == (4,)
    assert np.allclose(cn, cn_ref)

    pn = np.array([[1, 5, 18, 39]]) # shape = (1, 4)
    Un = np.array([[0.8, 0.8, 0.8]]) # shape = (1, 3)
    cn_ref = np.array([1., 6.25, 28.125, 76.171875])
    cn = _compute_cn(pn, Un, method='cum')
    assert cn.shape == (4,)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn
    pn = np.array([[1, 0, 18, 0, 39]])
    Un = np.array([[0.8, 0.8, 0.8, 0.8]])
    cn_ref = np.array([1., 0., 28.125, 0, 95.21484375])
    cn = _compute_cn(pn, Un, method='single')
    print cn
    assert cn.shape == (5,)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn
    pn = np.array([[1, 0, 18, 0, 39]])
    Un = np.array([[0.8, 0.8, 0.8, 0.8]])
    cn_ref = np.array([1., 0., 28.125, 0, 95.21484375])
    cn = _compute_cn(pn, Un, method='cum')
    assert cn.shape == (5,)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn
    pn = np.array([[10, 32,  0,  0],  # subrun 1
                   [ 4, 27, 56,  0],  # subrun 2
                   [ 1,  5, 18, 39]]) # subrun 3
    Un = np.array([[0.5,  0.,  0.],  # subrun 1
                   [0.6, 0.6,  0.],  # subrun 2
                   [0.8, 0.8, 0.8]]) # subrun 3
    cn_ref = np.array([1., 7.804878048780487, 28.592127505433467, 77.43701199388231])
    cn = _compute_cn(pn, Un, method='cum')
    assert cn.shape == (4,)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn_v
    pn = np.array([[[1, 3], [5, 6]]]) # shape = (1, 2, 2)
    Un = np.array([[0.8]]) # shape = (1, 1)
    cn_ref = np.array([[1., 1.], [6.25, 2.5]]) # shape = (2, 2)
    cn = _compute_cn_v(pn, Un, method='single')
    assert cn.shape == (2, 2)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn_v
    pn = np.array([[[1, 3], [5, 6]]]) # shape = (1, 2, 2)
    Un = np.array([[0.8]]) # shape = (1, 1)
    cn_ref = np.array([[1., 1.], [6.25, 2.5]]) # shape = (2, 2)
    cn = _compute_cn_v(pn, Un, method='cum')
    assert cn.shape == (2, 2)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn_v
    pn = np.array([[[1, 3], [5, 6], [10, 8]]]) # shape = (1, 3, 2)
    Un = np.array([[0.8, 0.8]]) # shape = (1, 2)
    cn_ref = np.array([[1., 1.], [6.25, 2.5], [15.625, 4.166666666666667]]) # shape = (3, 2)
    cn = _compute_cn_v(pn, Un, method='single')
    assert cn.shape == (3, 2)
    assert np.allclose(cn, cn_ref)

    ### test _compute_cn_v
    pn = np.array([[[1, 3], [5, 6], [10, 8]]]) # shape = (1, 3, 2)
    Un = np.array([[0.8, 0.8]]) # shape = (1, 2)
    cn_ref = np.array([[1., 1.], [6.25, 2.5], [15.625, 4.166666666666667]]) # shape = (3, 2)
    cn = _compute_cn_v(pn, Un, method='cum')
    assert cn.shape == (3, 2)
    assert np.allclose(cn, cn_ref)


    # results = [{'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}]
    # add_cn_to_results(results, 'Paul')
    # assert len(results) == 1
    # assert np.array_equal(results[0]['Paul']['pn'], np.array([1, 5, 18, 39]))
    # assert results[0]['metadata']['U'] == 0.8
    # assert results[0]['Paul']['cn'].shape == (4,)
    # # print results[0]['Paul']['cn']
    # assert np.allclose(results[0]['Paul']['cn'], np.array([1., 6.25, 28.125, 76.171875]))

    # ### test add_cn_to_results
    # ### should work with extra axes
    # results = [{'Paul': {'pn': np.array([[1, 3], [5, 6]])}, 'metadata': {'U': 0.8}}]
    # add_cn_to_results(results, 'Paul')
    # assert len(results) == 1
    # assert np.array_equal(results[0]['Paul']['pn'], np.array([[1, 3], [5, 6]]))
    # assert results[0]['metadata']['U'] == 0.8
    # assert results[0]['Paul']['cn'].shape == (2, 2)
    # # print results[0]['Paul']['cn']
    # assert np.allclose(results[0]['Paul']['cn'], np.array([[1., 1.], [6.25, 2.5]]))

    # ### test add_cn_to_results
    # ### should work with extra axes
    # results = [{'Paul': {'pn': np.array([[1, 3], [5, 6], [10, 8]])}, 'metadata': {'U': 0.8}}]
    # add_cn_to_results(results, 'Paul')
    # assert len(results) == 1
    # assert np.array_equal(results[0]['Paul']['pn'], np.array([[1, 3], [5, 6], [10, 8]]))
    # assert results[0]['metadata']['U'] == 0.8
    # assert results[0]['Paul']['cn'].shape == (3, 2)
    # # print results[0]['Paul']['cn']
    # assert np.allclose(results[0]['Paul']['cn'], np.array([[1., 1.], [6.25, 2.5], [15.625, 4.166666666666667]]))

    # ### test add_cn_to_results
    # run1 = {'Paul': {'pn': np.array([10, 32]), 'cn': np.array([1., 6.4])}, 'metadata': {'U': 0.5}}
    # run2 = {'Paul': {'pn': np.array([4, 27, 56]), 'cn': np.array([1., 6.4, 22.123456790123456])}, 'metadata': {'U': 0.6}}
    # run3 = {'Paul': {'pn': np.array([1, 5, 18, 39])}, 'metadata': {'U': 0.8}}
    # results = [run1, run2, run3]
    # add_cn_to_results(results, 'Paul')
    # assert len(results) == 3
    # assert np.array_equal(results[-1]['Paul']['pn'], np.array([1, 5, 18, 39]))
    # assert results[-1]['metadata']['U'] == 0.8
    # assert results[-1]['Paul']['cn'].shape == (4,)
    # assert np.allclose(results[-1]['Paul']['cn'], np.array([1., 6.4, 22.123456790123456, 59.917695473251015]))

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
