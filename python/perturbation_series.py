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
    # pn: (n, n)-array
    # sn: (n, n, m1, ...)-array
    # U: single value
    # return: (n, m1, ...)-array

    cn = np.zeros(sn.shape[1:])
    on = np.zeros(sn.shape[1:], dtype=complex)

    for k in range(0, sn.shape[0]):

        if (k==0):
            cn[0, ...] = c0
        else:
            if pn[k, k-1] != 0:
                cn[k, ...] = cn[k-1, ...] * pn[k, k] / (pn[k, k-1] * U)
            else:
                cn[k, ...] = cn[k-2, ...] * pn[k, k] / (pn[k, k-2] * U * U)

    on = cn * np.rollaxis(np.diagonal(sn), -1, 0) # np.diagonal sends the diagonalised axis to the left end

    return on


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

