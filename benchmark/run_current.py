from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
import numpy as np

alpha = 0.0
gamma = 0.5
epsilon_d = 0.
beta = 200.
U_qmc = 2.
n_cycles = 20000
n_warmup_cycles = 100
length_cycle = 1
random_seed = 15258
max_order = 2
p_dbl = 0
Nt_gf0 = 25000
tmax_gf0 = 200.0
tmax = 20.
muL = 0.
muR = 0.
random_seed = 15258

g0_lesser, g0_greater = make_g0_semi_circular(beta=beta, Gamma=gamma*gamma,
                                              tmax_gf0=tmax_gf0, Nt_gf0=Nt_gf0,
                                              epsilon_d=epsilon_d,
                                              muL=muL, muR=muR)

on_list, cn_list, on_last = [], [], []

S = SolverCore(g0_lesser, g0_greater)

for order in range(0, max_order):

    pn, sn = S.solve(U=U_qmc,
                     max_perturbation_order=order,
                     min_perturbation_order=0,
                     p_dbl=p_dbl,
                     tmax=tmax,
                     alpha=alpha,
                     verbosity=0,
                     n_cycles=n_cycles,
                     n_warmup_cycles=n_warmup_cycles,
                     random_seed=random_seed,
                     length_cycle=length_cycle)

    print "----------- order = ", order, "---------------"
    print " pn = ", pn
    print " sn = ", sn

    inv_U_qmc = 1. / U_qmc
    cn_over_Zqmc = np.array([x * inv_U_qmc ** n for n, x in enumerate(pn)])
    fact = cn_over_Zqmc[-2] / c_norm if order > 0 else 1
    cn = cn_over_Zqmc / fact
    on = cn * sn
    on_list.append(on)
    cn_list.append(cn)
    on_last.append(on[-1])
    c_norm = cn[-1]  # for next iter

    if mpi.rank == 0:
        print "----------- order = ", order, "---------------"
        print "cn = ", cn
        print "sn = ", sn
        print "on = ", on
        print "on_last = ", on_last

if mpi.rank == 0:
    print "--------- on ---------"
    print on_list
