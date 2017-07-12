import sys
sys.path.append('/home/bertrand/build/triqs/INSTALL_DIR/lib/python2.7/dist-packages')
sys.path.append('/home/bertrand/build/ctint_keldysh/python')

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

def generate_a_list(n):
    output = []
    for k in range(2**n):
        a = []
        i = 2**(n-1)
        sign = 1
        while i != 0:
            if k & i:
                a.append(1)
                sign = - sign
            else:
                a.append(0)
            i >>= 1
        output.append((sign, tuple(a)))
    return output


class KeldyshSumDet4 :

    def __init__(self, g0, tau, taup, alpha=0., use_cache=False):
        self.g0 = g0
        self.tau = tau
        self.taup = taup
        self.alpha = alpha
        self.matrix1 = np.empty((5, 5), dtype=complex)
        self.matrix2 = np.empty((4, 4), dtype=complex)
        self.cache = {}
        self.cache_used = 0
        self.use_cache = use_cache
        self.a_list = generate_a_list(4)

    def _det4(self, u, a):
        u1, u2, u3, u4 = u
        a1, a2, a3, a4 = a
        x = [(u1, a1), (u2, a2), (u3, a3), (u4, a4), self.tau]
        y = [(u1, a1), (u2, a2), (u3, a3), (u4, a4), self.taup]
        for i in range(5):
            for j in range(5):
                g = self.g0(x[i], y[j])
                self.matrix1[i, j] = g
                if (i<4) and (j<4):
                    self.matrix2[i, j] = g
                    if i == j:
                        self.matrix2[i, j] -= 1j * self.alpha
                        self.matrix1[i, j] -= 1j * self.alpha

        return linalg.det(self.matrix1) * linalg.det(self.matrix2)

    def __call__(self, u):
        """u = (u1, u2, u3, u4)"""

        if self.use_cache and u in self.cache:
            self.cache_used += 1
            return self.cache[u]
        else:
            output = 0.
            for sign, a in self.a_list:
                output += sign * self._det4(u, a)

            if self.use_cache:
                self.cache[u] = output

            return output

######################### integration methods ######################

def my_nquad(integrand, range_0, n, points):
    opts = {'epsrel': 1e-2, 'points': points, 'limit': 25}

    def range_f(*args):
        if len(args) > 0:
            return [range_0[0], args[0]]
        else:
            return range_0

    ranges = []
    for i in range(n):
        ranges.append(range_f)

    print 'real'
    rv, re = integrate.nquad(lambda *args: integrand(args).real,
                             ranges=ranges,
                             opts=opts)

    print 'imag'
    iv, ie = integrate.nquad(lambda *args: integrand(args).imag,
                             ranges=ranges,
                             opts=opts)

    return complex(rv, iv), complex(re, ie)


###############################################

def analytic_order4(g0_lesser, g0_greater, alpha, interaction_start, times):
    g0_keldysh = GKeldysh(g0_lesser, g0_greater)
    t2 = 0.0
    tmax = max(t2, np.max(times))

    world = MPI.COMM_WORLD
    times_split = np.array_split(times, world.size)
    if world.rank == 0:
        times_local = world.scatter(times_split)
    else:
        times_local = world.scatter(None)

    o4_less       = np.empty((len(times_local),), dtype=complex)
    o4_less_error = np.empty((len(times_local),), dtype=complex)

    for i, t in enumerate(times_local):

        print '[rank '+str(world.rank)+']', i+1, '/', len(times_local)

        integrand_lesser = KeldyshSumDet4(g0_keldysh, (t, 0), (t2, 1), alpha, use_cache=False)

        value, error = my_nquad(integrand_lesser, [-interaction_start, tmax], 4, [t])
        print "cache used:", integrand_lesser.cache_used
        del integrand_lesser
        o4_less[i], o4_less_error[i] = value, error

    o4_less       = np.concatenate(world.allgather(o4_less))
    o4_less_error = np.concatenate(world.allgather(o4_less_error))

    return (o4_less, o4_less_error)


if __name__ == '__main__':
    from ctint_keldysh import make_g0_semi_circular, make_g0_flat_band
    from pytriqs.archive import *
    import datetime
    world = MPI.COMM_WORLD

    # ### test my_nquad
    # v, e = my_nquad(lambda a: 1.+2.j, [-1., 1.], 3, [0.5])
    # print v, e
    # print (8. + 16.j) / 6.
    # assert abs(v - (8. + 16.j) / 6.) < abs(e)
    # assert abs(e) < 1e-12
    # print

    # ### test my_nquad
    # v, e = my_nquad(lambda a: 1. + a[2] + a[0]*a[1]**2, [0, 2], 3, [])
    # print v, e
    # print 8./6.+2.+2**6/60.
    # assert abs(v - (8./6.+2.+2**6/60.)) < abs(e)
    # assert abs(e) < 1e-12
    # print

    # ###
    # print generate_a_list(4)


    # exit()

    #-----------------
    starttime = datetime.datetime.now()
    times = np.linspace(-40.0, -20.0, 4)
    Nt_gf0 = 200
    g0_lesser, g0_greater = make_g0_flat_band(beta=200.,
                                              Gamma=0.2,
                                              epsilon_d=0.,
                                              muL=0.,
                                              muR=0.,
                                              tmax_gf0=250.,
                                              Nt_gf0=Nt_gf0)

    g0_lesser_sym = g0_lesser.copy()
    for i, t in enumerate(g0_lesser.mesh):
        t = t.real
        g0_lesser_sym.data[i] = 0.5 * (g0_lesser(t) - np.conjugate(g0_lesser(-t)))

    g0_greater_sym = g0_greater.copy()
    for i, t in enumerate(g0_greater.mesh):
        t = t.real
        g0_greater_sym.data[i] = 0.5 * (g0_greater(t) - np.conjugate(g0_greater(-t)))

    g0_lesser_sym.data[Nt_gf0-1][0, 0] = 0.5j
    g0_greater_sym.data[Nt_gf0-1][0, 0] = -0.5j

    interaction_start = 100.0
    o4_less = analytic_order4(g0_lesser, g0_greater, 0.5, interaction_start, times)

    deltatime = datetime.datetime.now() - starttime
    if world.rank == 0:
        print 'Run time = ', deltatime

        with HDFArchive("order4_params3.new.ref.h5", 'w') as ar:
            ar["times"] = times
            ar["interaction_start"] = interaction_start
            ar["o4_less"] = o4_less[0]
            ar["o4_less_error"] = o4_less[1]

