import numpy as np

def perturbation_series(_c0, pn, sn, U):
    # TODO: sanity checks
    # _c0: single value or (m1, ...)-array
    # pn: (n,)-array
    # sn: (n, m1, ...)-array
    # U: single value
    # return: (n, m1, ...)-array

    # transforms c0 in array if it si a single value
    c0 = np.array(_c0)
    if c0.size == 1:
        c0 = c0 * np.ones(sn.shape[1:])

    p_is_zero = pn == 0
    pn[p_is_zero] = 1 # avoids division by zero, works with any non zero value

    cn = np.zeros(sn.shape)
    on = np.zeros(sn.shape, dtype=complex)

    for k in range(0, sn.shape[0]):

        if (k==0):
            cn[0, ...] = c0
        else:
            cn[k, ...] = cn[k-1, ...] * pn[k] / (pn[k-1] * U)

    pn[p_is_zero] = 0 # get back to original pn

    on = cn * sn
    return on


def perturbation_series_errors(c0, pn, sn, U, c0_error, pn_error, sn_error):
    # TODO: sanity checks
    # c0: single value
    # pn: (n,)-array
    # sn: (n, m)-array
    # U: single value
    # c0_error: single value
    # pn_error: (n,)-array
    # sn_error: (n, m)-array
    # return: ((n, m)-array, (n, m)-array)

    n, m = sn.shape

    pn_rel_error = pn_error / pn

    cn = np.zeros((n,))
    on = np.zeros((n, m), dtype=complex)
    cn_rel_error = np.zeros((n,))
    cn_error = np.zeros((n,))
    on_error = np.zeros((n, m))


    for k in range(0, n):

        if (k==0):
            cn[0] = c0
            cn_rel_error[0] = c0_error / c0
        else:
            cn[k] = cn[k-1] * pn[k] / (pn[k-1] * U)
            cn_rel_error[k] = cn_rel_error[k-1] + pn_rel_error[k] + pn_rel_error[k-1]

    cn_error = cn * cn_rel_error

    on = cn[:, np.newaxis] * sn
    on_error = np.abs(sn) * cn_error[:, np.newaxis ]+ cn[:, np.newaxis] * sn_error

    return on, on_error


def staircase_perturbation_series(c0, pn, sn, U):
    # TODO: sanity checks
    # c0: single value
    # pn: list of 1D arrays
    # sn: list of ND arrays
    # U: list
    # return: N+1D array

    assert(len(pn) > 1)
    assert(len(sn) == len(pn))
    assert(len(U) >= len(pn) - 1)
    for i in range(len(pn)-1):
        assert(len(pn[i+1]) > len(pn[i]))
        assert(len(sn[i]) == len(pn[i]))
    assert(len(sn[-1]) == len(pn[-1]))

    nb_orders = len(pn[-1])
    cn = np.zeros((nb_orders,), dtype=complex)
    on = np.zeros(sn[-1].shape, dtype=complex)

    # fill in cn
    cn[0] = c0
    k = 1
    for i in range(len(pn)):

        while k < len(pn[i]):
            if pn[i][k-1] != 0:
                cn[k] = cn[k-1] * pn[i][k] / (pn[i][k-1] * U[i-1])
            else:
                cn[k] = cn[k-2] * pn[i][k] / (pn[i][k-2] * U[i-1] * U[i-1])
            k += 1

    # fill in on
    k = 0
    for i in range(len(sn)):
        while k < len(sn[i]):
            on[k, ...] = cn[k] * sn[i][k, ...]
            k += 1

    return on, cn

def check_staircase(staircase_list):
    if len(staircase_list) == 0:
        return
    assert(len(staircase_list[0]) > 0) # no empty array
    for i in range(len(staircase_list)-1):
        assert(len(staircase_list[i+1]) > len(staircase_list[i])) # TODO: >=


def compute_cn(pn, U, c0=1.):
    assert(len(pn) >= 1)
    assert(len(pn) == len(U))
    assert(len(pn[0]) > 1) # order zero not in pn
    for i in range(len(pn)-1):
        assert(len(pn[i+1]) > len(pn[i])) # TODO: >=

    cn = np.zeros((len(pn[-1]),))
    cn[0] = c0
    k = 1
    for i in range(len(pn)):

        while k < len(pn[i]):
            if pn[i][k-1] != 0:
                cn[k] = cn[k-1] * pn[i][k] / (pn[i][k-1] * U[i])
            else:
                cn[k] = cn[k-2] * pn[i][k] / (pn[i][k-2] * U[i] * U[i])
            k += 1

    return cn

def staircase_leading_elts(staircase_list):
    check_staircase(staircase_list)
    output = []
    k = 0
    for array in staircase_list:
        while k < len(array):
            output.append(array[k])
            k += 1

    return np.array(output)


# TO BE VERIFIED
def staircase_perturbation_series_cum(c0, _pn, sn, U):
    # TODO: sanity checks
    # c0: single value
    # pn: (n, n)-array
    # sn: (n, n, m1, ...)-array
    # U: single value
    # return: (n, m1, ...)-array

    pn = _pn.copy()

    pn[0, 0] = 0
    pn[pn != pn] = 0
    sn[sn != sn] = 0

    pn_over_p0 = pn[1:, :] / pn[1:, 0:1]
    # Nn_summed = np.cumsum(pn[1:, :], axis=1)
    Nn_summed = pn[1:, :].copy()
    for k in range(1, pn.shape[1]):
        Nn_summed[k-1:, k] += Nn_summed[k-1:, k-1]

    print pn_over_p0
    print
    print Nn_summed
    print

    pn_equiv = np.average(pn_over_p0, weights=Nn_summed, axis=0)

    while pn.ndim != sn.ndim:
        pn = pn[..., np.newaxis]

    pn_sum = np.sum(pn, axis=0)
    print np.squeeze(pn_sum)
    pn_sum[pn_sum == 0] = 1 # avoids divide by zero

    sn_sum = np.sum(sn * pn, axis=0) / pn_sum

    return perturbation_series(c0, pn_equiv, sn_sum, U)

if __name__ == '__main__':

    # test compute_cn
    pn = [[10, 22, 45, 89]]
    U = [5]
    cn_ref = [1., 0.44, 0.18, 0.0712]
    cn = compute_cn(pn, U)
    assert(len(cn) == len(cn_ref))
    assert(np.allclose(cn, cn_ref))

    # test compute_cn
    pn = [[10, 0, 22, 0, 45, 0, 89]]
    U = [5]
    cn_ref = [1., 0., 0.088, 0., 0.0072, 0., 0.0005696]
    cn = compute_cn(pn, U)
    assert(len(cn) == len(cn_ref))
    assert(np.allclose(cn, cn_ref))

    # test compute_cn
    pn = [[10, 22],
          [ 4, 25, 45],
          [16, 32, 66, 89],
          [ 3, 28, 47, 70, 126, 252]]
    U = [5, 2, 0.4, 1.2]
    cn_ref = [1., 0.44, 0.396, 1.335, 2.0025, 3.3375]
    cn = compute_cn(pn, U)
    assert(len(cn) == len(cn_ref))
    assert(np.allclose(cn, cn_ref))

    # test compute_cn
    pn = [[10, 0, 22],
          [ 4, 0, 25, 0, 45],
          [16, 0, 32, 0, 66, 0, 89],
          [ 3, 0, 28, 0, 47, 0, 70, 0, 126, 0, 252]]
    U = [5, 2, 0.4, 1.2]
    cn_ref = [1., 0., 0.088, 0., 0.0396, 0., 0.33375, 0., 0.4171875, 0., 0.5794270833333333]
    cn = compute_cn(pn, U)
    assert(len(cn) == len(cn_ref))
    assert(np.allclose(cn, cn_ref))

