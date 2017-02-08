import numpy as np
from numpy import linalg
# from scipy import integrate
import mpmath
from copy import copy

class GKeldysh :

    def __init__(self, g0_lesser, g0_greater):
        self.g0_lesser = g0_lesser
        self.g0_greater = g0_greater

    def __call__(self, alpha1, alpha2):
        u1, a1 = alpha1
        u2, a2 = alpha2

        if u1 == u2 and a1 == a2:
            return self.g0_lesser(u1 - u2)[0, 0]
        
        if a1 == a2:
            is_greater = (a1 != (u1 > u2))
        else:
            is_greater = a1
            
        if is_greater:
            return self.g0_greater(u1 - u2)[0, 0]
        else:
            return self.g0_lesser(u1 - u2)[0, 0]


class KeldyshSumDet2 :

    def __init__(self, g0, tau, taup):
        self.g0 = g0
        self.tau = tau
        self.taup = taup
        self.matrix1 = np.empty((3, 3), dtype=complex)
        self.matrix2 = np.empty((2, 2), dtype=complex)

    def _det2(self, u1, u2, a1, a2):
        x = [(u1, a1), (u2, a2), self.taup]
        y = [(u1, a1), (u2, a2), self.tau]
        for i in range(3):
            for j in range(3):
                g = self.g0(x[i], y[j])
                self.matrix1[i, j] = g
                if (i<2) and (j<2):
                    self.matrix2[i, j] = g
        return linalg.det(self.matrix1) * linalg.det(self.matrix2)

    def __call__(self, u1, u2):

        output = 0.
        for a1 in [0, 1]:
            for a2 in [0, 1]:
                output += pow(-1, a1+a2) * self._det2(u1, u2, a1, a2)

        return output


def analytic_order2(g0_lesser, g0_greater, interaction_start, times, t2=0.0):
    g0_keldysh = GKeldysh(g0_lesser, g0_greater)
    tmax = max(t2, np.max(times))
    o2_less       = np.empty((len(times),), dtype=complex)
    o2_less_error = np.empty((len(times),), dtype=complex)
    o2_grea       = np.empty((len(times),), dtype=complex)
    o2_grea_error = np.empty((len(times),), dtype=complex)

    for i, t in enumerate(times):

        if i % (1 + len(times) // 10) == 0:
            print i, '/', len(times)

        integrand_lesser = KeldyshSumDet2(g0_keldysh, (t, 1), (t2, 0))
        integrand_greater = KeldyshSumDet2(g0_keldysh, (t, 0), (t2, 1))

        v, e = mpmath.quadgl(integrand_lesser,
                           [-interaction_start, tmax], 
                           [-interaction_start, tmax],
                           # maxdegree=4,
                           error=True,
                           verbose=True)
        o2_less[i], o2_less_error[i] = v, e

        v, e= mpmath.quadgl(integrand_greater,
                          [-interaction_start, tmax], 
                          [-interaction_start, tmax], 
                           # maxdegree=4,
                          error=True,
                          verbose=True)
        o2_grea[i], o2_grea_error[i] = v, e

    return (o2_less, o2_less_error), (o2_grea, o2_grea_error)


if __name__ == '__main__':
    from ctint_keldysh import make_g0_semi_circular
    from pytriqs.archive import *
    mpmath.mp.dps = 5

    #-----------------
    times = np.linspace(-40.0, 0.0, 11)
    g0_lesser, g0_greater = make_g0_semi_circular(beta=200,
                                                  Gamma=0.5,
                                                  tmax_gf0=100.0,
                                                  Nt_gf0=25000,
                                                  epsilon_d=0.5,
                                                  muL=0.0,
                                                  muR=0.0)

    # (o2_less, o2_less_error), (o2_grea, o2_grea_error) = analytic_order2(g0_lesser, g0_greater, 40.0, times)
    o2_less, o2_grea = analytic_order2(g0_lesser, g0_greater, 40.0, times)

    with HDFArchive("test_o2_1.ref.h5", 'w') as ar:
        ar["times"] = times
        ar["t2"] = 0.0
        ar["o2_lesser"] = o2_less[0]
        ar["o2_lesser_error"] = o2_less[1]
        ar["o2_greater"] = o2_grea[0]
        ar["o2_greater_error"] = o2_grea[1]

