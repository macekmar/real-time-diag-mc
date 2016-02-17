from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
import numpy as np

beta = 200.
gamma = 0.5
tmax_gf0 = 100.0
Nt_gf0 = 25000
epsilon_d = 0.
muL = 0.
muR = 0.

U_qmc = 2.
<<<<<<< HEAD
n_cycles = 10000
n_warmup_cycles = 100
length_cycle = 10
random_seed = 15258
max_order = 3
=======
>>>>>>> single insert moves and move change work, give the same MC chain
p_dbl = 0
tmax = 10.
alpha = 0.0
random_seed = 15258
##n_cycles = 200000
##n_warmup_cycles = 100
##length_cycle = 10
#max_order = 3
n_cycles = 2000
n_warmup_cycles = 10
length_cycle = 1
max_order = 2

g0_lesser, g0_greater = make_g0_semi_circular(beta=beta, Gamma=gamma*gamma,
                                              tmax_gf0=tmax_gf0, Nt_gf0=Nt_gf0,
                                              epsilon_d=epsilon_d,
                                              muL=muL, muR=muR)

on_list, cn_list, on_last = [], [], []
sn_errors_last,pn_errors_last = [],[]

S = SolverCore(g0_lesser, g0_greater)

<<<<<<< HEAD
for order in range(max_order):

    (pn, sn),(pn_error,sn_error) = S.solve(U=U_qmc,
                     max_perturbation_order=order,
                     min_perturbation_order=0,
                     p_dbl=p_dbl,
                     tmax=tmax,
                     alpha=alpha,
                     verbosity=3,
                     n_cycles=n_cycles,
                     n_warmup_cycles=n_warmup_cycles,
                     random_seed=random_seed,
                     length_cycle=length_cycle)

    #We do not print this for every node
    #print "----------- order = ", order, "---------------"
    #print "You are on node ", mpi.rank
    #print " pn = ", pn
    #print " sn = ", sn
    #print " pn_error = ", pn_error
    #print " sn_error = ", sn_error

    inv_U_qmc = 1. / U_qmc
    cn_over_Zqmc = np.array([x * inv_U_qmc ** n for n, x in enumerate(pn)])
    fact = cn_over_Zqmc[-2] / c_norm if order > 0 else 1
    cn = cn_over_Zqmc / fact
    on = cn * sn
    on_list.append(on)
    cn_list.append(cn)
    on_last.append(on[-1])
    pn_errors_last.append(pn_error[-1]/pn[-1])
    sn_errors_last.append(sn_error[-1]/sn[-1]) #We take the normalized deviation

    c_norm = cn[-1]  # for next iter

    #print "c_norm = ", c_norm  

    if mpi.rank == 0:
        print "----------- order = ", order, "---------------"
        print "cn = ", cn
        print " pn = ", pn
        print "sn = ", sn
        print "sn normalized deviation = ", sn_errors_last
        print "pn/z normalized deviation = ", pn_errors_last
        print "on = ", on
        print "on_last = ", on_last
=======
c_norm = g0_lesser(0)[0,0].imag
print "Non interacting charge : ", c_norm 

qn_last, cn_last, sn_last, pn_last = [c_norm], [
    abs(c_norm)], [c_norm / abs(c_norm)], [1]

for Nmax in range(1, max_order + 1):
    if mpi.rank == 0:
        print "--------- Nmax = ", Nmax
    pn, sn = S.solve(U=U_qmc,
            max_perturbation_order=Nmax,
            p_dbl=p_dbl,
            tmax=tmax,
            alpha=alpha,
            verbosity=3,
            random_seed=random_seed,
            n_cycles=n_cycles,
            n_warmup_cycles=n_warmup_cycles,
            length_cycle=length_cycle)
    # Bare cn,sn coming from the simulations with an extra factor
    # (U_qmc)**n and
    sn_last.append(sn[-1])
    new_cn = cn_last[-1] / float(U_qmc) * pn[-1] / pn[-2]
    cn_last.append(new_cn)
    qn_last.append(new_cn * sn[-1])
    pn_last.append(pn[-1])

    if mpi.rank == 0:
        print "pn = ", pn
        print "sn = ", sn
        print " qn_last = ", qn_last
>>>>>>> single insert moves and move change work, give the same MC chain

if mpi.rank == 0:
    print "--------- SUMMARY ---------"
    print " qn_last = ", qn_last
    print " cn_last = ", cn_last
    print " sn_last = ", sn_last
    print " pn_last = ", pn_last

#for order in range(max_order):
#
#    pn, sn = S.solve(U=U_qmc,
#                     max_perturbation_order=order,
#                     min_perturbation_order=0,
#                     p_dbl=p_dbl,
#                     tmax=tmax,
#                     alpha=alpha,
#                     verbosity=3,
#                     random_seed=random_seed,
#                     n_cycles=n_cycles,
#                     n_warmup_cycles=n_warmup_cycles,
#                     length_cycle=length_cycle)
#
#    print "----------- order = ", order, "---------------"
#    print " pn = ", pn
#    print " sn = ", sn
#
#    inv_U_qmc = 1. / U_qmc
#    cn_over_Zqmc = np.array([x * inv_U_qmc ** n for n, x in enumerate(pn)])
#    fact = cn_over_Zqmc[-2] / c_norm if order > 0 else 1
#    cn = cn_over_Zqmc / fact
#    on = cn * sn
#    on_list.append(on)
#    cn_list.append(cn)
#    on_last.append(on[-1])
#    c_norm = cn[-1]  # for next iter
#
#    print "c_norm = ", c_norm  
#
#    if mpi.rank == 0:
#        print "----------- order = ", order, "---------------"
#        print "cn = ", cn
#        print "sn = ", sn
#        print "on = ", on
#        print "on_last = ", on_last
#
#if mpi.rank == 0:
#    print "--------- on ---------"
#    print on_list
