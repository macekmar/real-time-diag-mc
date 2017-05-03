import numpy as np
from numpy import linalg
from scipy import integrate
from copy import copy
from mpi4py import MPI

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

    def __init__(self, g0, tau, taup, alpha=0.):
        self.g0 = g0
        self.tau = tau
        self.taup = taup
        self.alpha = alpha
        self.matrix1 = np.empty((3, 3), dtype=complex)
        self.matrix2 = np.empty((2, 2), dtype=complex)
        self.cache = {}
        self.cache_used = 0

    def _det2(self, u1, u2, a1, a2):
        x = [(u1, a1), (u2, a2), self.tau]
        y = [(u1, a1), (u2, a2), self.taup]
        for i in range(3):
            for j in range(3):
                g = self.g0(x[i], y[j])
                self.matrix1[i, j] = g
                if (i<2) and (j<2):
                    self.matrix2[i, j] = g
                    if i == j:
                        self.matrix2[i, j] -= 1j * self.alpha
                        self.matrix1[i, j] -= 1j * self.alpha

        return linalg.det(self.matrix1) * linalg.det(self.matrix2)

    def __call__(self, u1, u2):

        if (u1, u2) in self.cache:
            self.cache_used += 1
            return self.cache[(u1, u2)]
        else:
            output = 0.
            for a1 in [0, 1]:
                for a2 in [0, 1]:
                    output += pow(-1, a1+a2) * self._det2(u1, u2, a1, a2)
            self.cache[(u1, u2)] = output

            return output


def my_dblquad(integrand, times_list):
    rv, iv = 0., 0.
    re, ie = 0., 0.

    for i1, t1 in enumerate(times_list[:-1]):
        for i2, t2 in enumerate(times_list[:-1]):
            t2p = times_list[i2+1]
            v, e = integrate.dblquad(lambda u1, u2 : integrand(u1, u2).real,
                                     t1,
                                     times_list[i1+1],
                                     lambda x : t2,
                                     lambda x : t2p,
                                     epsabs=1.e-4,
                                     epsrel=1.e-4)
            rv += v
            re += e
            v, e = integrate.dblquad(lambda u1, u2 : integrand(u1, u2).imag,
                                     t1,
                                     times_list[i1+1],
                                     lambda x : t2,
                                     lambda x : t2p,
                                     epsabs=1.e-4,
                                     epsrel=1.e-4)
            iv += v
            ie += e

    return complex(rv, iv), complex(re, ie)


def analytic_order2(g0_lesser, g0_greater, alpha, interaction_start, times):
    g0_keldysh = GKeldysh(g0_lesser, g0_greater)
    t2 = 0.0
    tmax = max(t2, np.max(times))

    world = MPI.COMM_WORLD
    times_split = np.array_split(times, world.size)
    if world.rank == 0:
        times_local = world.scatter(times_split)
    else:
        times_local = world.scatter(None)

    o2_less       = np.empty((len(times_local),), dtype=complex)
    o2_less_error = np.empty((len(times_local),), dtype=complex)
    o2_grea       = np.empty((len(times_local),), dtype=complex)
    o2_grea_error = np.empty((len(times_local),), dtype=complex)

    for i, t in enumerate(times_local):

        if world.rank == 0:
            if (i * world.size) % (1 + len(times) // 10) == 0:
                print (i * world.size), '/', len(times)

        integrand_lesser  = KeldyshSumDet2(g0_keldysh, (t, 0), (t2, 1), alpha)
        integrand_greater = KeldyshSumDet2(g0_keldysh, (t, 1), (t2, 0), alpha)

        value, error = my_dblquad(integrand_lesser, [-interaction_start, t, tmax])
        # print "cache used:", integrand_lesser.cache_used
        o2_less[i], o2_less_error[i] = -0.5 * value, 0.5 * error

        value, error = my_dblquad(integrand_greater, [-interaction_start, t, tmax])
        # print "cache used:", integrand_greater.cache_used
        o2_grea[i], o2_grea_error[i] = -0.5 * value, 0.5 * error

    o2_less       = np.concatenate(world.allgather(o2_less))
    o2_less_error = np.concatenate(world.allgather(o2_less_error))
    o2_grea       = np.concatenate(world.allgather(o2_grea))
    o2_grea_error = np.concatenate(world.allgather(o2_grea_error))

    return (o2_less, o2_less_error), (o2_grea, o2_grea_error)


if __name__ == '__main__':
    from ctint_keldysh import make_g0_semi_circular, make_g0_flat_band
    from pytriqs.archive import *
    import datetime
    world = MPI.COMM_WORLD

    #-----------------
    starttime = datetime.datetime.now()
    times = np.linspace(-40.0, 0.0, 50)
    g0_lesser, g0_greater = make_g0_semi_circular(beta=200,
                                                  Gamma=0.5,
                                                  epsilon_d=0.5,
                                                  muL=0.0,
                                                  muR=0.0,
                                                  tmax_gf0=100.0,
                                                  Nt_gf0=25000)

    interaction_start = 40.0
    o2_less, o2_grea = analytic_order2(g0_lesser, g0_greater, 0.0, interaction_start, times)

    deltatime = datetime.datetime.now() - starttime
    if world.rank == 0:
        print 'Run time = ', deltatime

    with HDFArchive("order2_params1.ref.h5", 'w') as ar:
        ar.create_group("less")
        less = ar["less"]
        less["times"] = times
        less["interaction_start"] = interaction_start
        less["o2"] = o2_less[0]
        less["o2_error"] = o2_less[1]

        ar.create_group("grea")
        grea = ar["grea"]
        grea["times"] = times
        grea["interaction_start"] = interaction_start
        grea["o2"] = o2_grea[0]
        grea["o2_error"] = o2_grea[1]

    #-----------------
    starttime = datetime.datetime.now()
    times = np.linspace(-100.0, 0.0, 50)
    g0_lesser, g0_greater = make_g0_flat_band(beta=200.,
                                              Gamma=0.2,
                                              epsilon_d=0.,
                                              muL=0.,
                                              muR=0.,
                                              tmax_gf0=250.,
                                              Nt_gf0=10000)

    g0_lesser_sym = g0_lesser.copy()
    for i, t in enumerate(g0_lesser.mesh):
        t = t.real
        g0_lesser_sym.data[i] = 0.5 * (g0_lesser(t) - np.conjugate(g0_lesser(-t)))

    g0_greater_sym = g0_greater.copy()
    for i, t in enumerate(g0_greater.mesh):
        t = t.real
        g0_greater_sym.data[i] = 0.5 * (g0_greater(t) - np.conjugate(g0_greater(-t)))

    g0_lesser_sym.data[10000-1][0, 0] = 0.5j
    g0_greater_sym.data[10000-1][0, 0] = -0.5j

    interaction_start = 150.0
    o2_less, o2_grea = analytic_order2(g0_lesser, g0_greater, 0.5, interaction_start, times)

    deltatime = datetime.datetime.now() - starttime
    if world.rank == 0:
        print 'Run time = ', deltatime

    with HDFArchive("order2_params3.ref.h5", 'w') as ar:
        ar.create_group("less")
        less = ar["less"]
        less["times"] = times
        less["interaction_start"] = interaction_start
        less["o2"] = o2_less[0]
        less["o2_error"] = o2_less[1]

        ar.create_group("grea")
        grea = ar["grea"]
        grea["times"] = times
        grea["interaction_start"] = interaction_start
        grea["o2"] = o2_grea[0]
        grea["o2_error"] = o2_grea[1]

