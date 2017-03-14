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

    p_is_zero = (pn == 0)
    pn[p_is_zero] = 1 # avoids division by zero, works with any non zero value

    cn = np.zeros(sn.shape[1:])
    on = np.zeros(sn.shape[1:], dtype=complex)

    for k in range(0, sn.shape[0]):

        if (k==0):
            cn[0, ...] = c0
        else:
            cn[k, ...] = cn[k-1, ...] * pn[k, k] / (pn[k, k-1] * U)

    pn[p_is_zero] = 0 # get back to original pn

    on = cn * np.rollaxis(np.diagonal(sn), -1, 0) # np.diagonal sends the diagonalised axis to the left end

    return on

