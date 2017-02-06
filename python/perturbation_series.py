import numpy as np

def perturbation_series(c0, pn, sn, U):
    # TODO: sanity checks

    n, m = sn.shape
    
    c0 = np.array(c0)
    if len(c0) == 1:
        c0 = c0 * np.ones((m,))

    cn = np.zeros((n, m))
    on = np.zeros((n, m), dtype=complex)

    for k in range(0, n):

        if (k==0):
            cn[0, :] = c0
        else:
            cn[k, :] = cn[k-1, :] * pn[k] / (pn[k-1] * U)

    on = cn * sn
    return on


def perturbation_series_errors(c0, pn, sn, U, c0_error, pn_error, sn_error):
    # TODO: sanity checks
    # works only for a single c0

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


def stairwise_perturbation_series(c0, pn, sn, U):
    # TODO: sanity checks

    n, m = pn.shape()
    pn_eff = np.zeros((n,))

    pn_eff[0] = 1.0 # any value works
    for k in range(1, n):
        pn_eff[k] = pn_eff[k-1] * pn[k, k] / pn[k, k-1]

    return perturbation_series(c0, pn_eff, np.diagonal(sn), U)

