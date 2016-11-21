from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import time

# Parameters of the model
alpha = 0.0
gamma = 0.5
epsilon_d = 0.
beta = 200.
U_qmc = 2.
n_cycles = 5000#00
n_warmup_cycles = 1000#50000
length_cycle = 1#50
max_order = 15
p_dbl = 0
Nt_gf0 = 25000
tmax_gf0 = 100.0
tmax = 10.
#V = muL-muR
muL = 0.
muR = 0.

g0_lesser, g0_greater = make_g0_semi_circular(beta=beta, Gamma=gamma * gamma,
                                              tmax_gf0=tmax_gf0, Nt_gf0=Nt_gf0,
                                              epsilon_d=epsilon_d,
                                              muL=muL, muR=muR)

S = SolverCore(g0_lesser, g0_greater)

# The final lists
on_list, cn_last, on_last, sn_last = [], [], [], [] # Final values
sn_errors_last, pn_errors_last = [], []             # Final errors
on_relative_errors = []
duration = []

for order in range(0, max_order):

    if mpi.rank == 0:
        print
        print "----------- order = ", order, "---------------"

    (pn, sn), (pn_error, sn_error) = S.solve(U=U_qmc,
                                             op_to_measure=[[(0, tmax, 0), (0, tmax, 1)], []],
                                             max_perturbation_order=order,
                                             min_perturbation_order=0,
                                             p_dbl=p_dbl,
                                             alpha=alpha,
                                             n_cycles=n_cycles,
                                             n_warmup_cycles=n_warmup_cycles,
                                             length_cycle=length_cycle)

    duration.append(S.solve_duration)

    inv_U_qmc = 1. / U_qmc
    cn_over_Zqmc = np.array([x * inv_U_qmc ** n for n, x in enumerate(pn)])
    fact = cn_over_Zqmc[-2] / c_norm if order > 0 else 1
    cn = cn_over_Zqmc / fact
    on = cn * sn
    on_list.append(on)

    # Saving the values
    cn_last.append(cn[-1])
    on_last.append(on[-1])
    sn_last.append(sn[-1])

    # Saving the errors
    # This is the same one as the cn normalized deviation.
    pn_errors_last.append(pn_error[-1] / pn[-1])
    # We take the normalized deviation
    sn_errors_last.append(sn_error[-1] / np.abs(sn[-1]))
    on_relative_errors.append(pn_errors_last[-1] + sn_errors_last[-1])

    c_norm = cn[-1]  # for next iter

    if mpi.rank == 0:
        print "cn = ", cn
        print " pn = ", pn
        print "sn = ", sn
        print "sn normalized deviation = ", sn_errors_last
        print "pn/z normalized deviation = ", pn_errors_last
        print "on = ", on
        print "on_last = ", on_last

if mpi.rank == 0:
    print "--------- on ---------"
    print on_last
    print "-------Relative error----"
    print on_relative_errors
    # We save the results
    with HDFArchive('final_results.h5', 'a') as ar:  # A file to store the results
        ar['on_values'] = on_last
        ar['on_errors'] = on_relative_errors
        ar['cn_values'] = cn_last
        ar['cn_errors'] = pn_errors_last
        ar['sn_values'] = sn_last
        ar['sn_errors'] = sn_errors_last
        ar['duration'] = duration
