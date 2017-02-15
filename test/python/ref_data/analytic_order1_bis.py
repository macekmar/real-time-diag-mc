import numpy as np
from scipy import integrate
# import mpmath
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


class KeldyshSumDet1 :

    def __init__(self, g0, tau, taup):
        self.g0 = g0
        self.g0_ttp = self.g0(tau, taup)
        self.tau = tau
        self.taup = taup
        self.cache = {}
        self.cache_used = 0

    def _det(self, u, a):

        g0_aa = self.g0((u, a), (u, a))
        g0_atp = self.g0((u, a), self.taup)
        g0_ta = self.g0(self.tau, (u, a))

        return (g0_aa * self.g0_ttp - g0_atp * g0_ta) * g0_aa

    def __call__(self, u):

        if u in self.cache:
            self.cache_used += 1
            return self.cache[u]
        else:
            output = 0.
            for a in [0, 1]:
                output += pow(-1, a) * self._det(u, a)
            self.cache[u] = output

            return output


def analytic_order1(g0_lesser, g0_greater, interaction_start, times, t2=0.0):
    g0_keldysh = GKeldysh(g0_lesser, g0_greater)
    tmax = max(t2, np.max(times))
    o1_less       = np.empty((len(times),), dtype=complex)
    o1_less_error = np.empty((len(times),), dtype=complex)
    o1_grea       = np.empty((len(times),), dtype=complex)
    o1_grea_error = np.empty((len(times),), dtype=complex)

    for i, t in enumerate(times):

        if i % (1 + len(times) // 10) == 0:
            print i, '/', len(times)

        integrand_lesser = KeldyshSumDet1(g0_keldysh, (t, 0), (t2, 1))
        integrand_greater = KeldyshSumDet1(g0_keldysh, (t, 1), (t2, 0))

        rv, re = integrate.quad(lambda u : integrand_lesser(u).real,
                                 -interaction_start,
                                 tmax,
                                 epsabs=1.e-04, 
                                 epsrel=1.e-04)

        iv, ie = integrate.quad(lambda u : integrand_lesser(u).imag,
                                 -interaction_start,
                                 tmax,
                                 epsabs=1.e-04, 
                                 epsrel=1.e-04)

        # print "cache used:", integrand_lesser.cache_used
        o1_less[i], o1_less_error[i] = 1j * complex(rv, iv), complex(ie, re)

        rv, re = integrate.quad(lambda u : integrand_greater(u).real,
                                 -interaction_start,
                                 tmax,
                                 epsabs=1.e-04, 
                                 epsrel=1.e-04)

        iv, ie = integrate.quad(lambda u : integrand_greater(u).imag,
                                 -interaction_start,
                                 tmax,
                                 epsabs=1.e-04, 
                                 epsrel=1.e-04)

        # print "cache used:", integrand_greater.cache_used
        o1_grea[i], o1_grea_error[i] = 1j * complex(rv, iv), complex(ie, re)

    return (o1_less, o1_less_error), (o1_grea, o1_grea_error)


if __name__ == '__main__':
    from ctint_keldysh import make_g0_semi_circular
    from pytriqs.archive import *
    # mpmath.mp.dps = 5

    #-----------------
    times = np.linspace(-40.0, 0.0, 101)
    g0_lesser, g0_greater = make_g0_semi_circular(beta=200,
                                                  Gamma=0.5,
                                                  epsilon_d=0.5,
                                                  muL=0.0,
                                                  muR=0.0,
                                                  tmax_gf0=100.0,
                                                  Nt_gf0=25000)

    # (o1_less, o1_less_error), (o1_grea, o1_grea_error) = analytic_order1(g0_lesser, g0_greater, 40.0, times)
    interaction_start = 40.0
    o1_less, o1_grea = analytic_order1(g0_lesser, g0_greater, interaction_start, times)

    with HDFArchive("order1_params1_bis.ref.h5", 'w') as ar:
        ar.create_group("less")
        less = ar["less"]
        less["times"] = times
        less["interaction_start"] = interaction_start
        less["o1_less"] = o1_less[0]
        less["o1_less_error"] = o1_less[1]

        ar.create_group("grea")
        grea = ar["grea"]
        grea["times"] = times
        grea["interaction_start"] = interaction_start
        grea["o1_grea"] = o1_grea[0]
        grea["o1_grea_error"] = o1_grea[1]

