from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
from pytriqs.archive import *
import numpy as np
import time


#Parameters of the model
alpha = 0.0
gamma = 0.5
epsilon_d = 0.
beta = 200.
U_qmc = 2.
n_cycles = 200000
n_warmup_cycles = 10000
length_cycle = 10
max_order = 3
p_dbl = 0
Nt_gf0 = 25000
tmax_gf0 = 100.0
tmax = 10.
#V = muL-muR
muL = 0.
muR = 0.

#Picking the random seeds, one for each core
RS_1 = 213345
RS_2 = 456465
random_seed = RS_1 + mpi.rank * RS_2 


g0_lesser, g0_greater = make_g0_semi_circular(beta=beta, Gamma=gamma*gamma,

                                              tmax_gf0=tmax_gf0, Nt_gf0=Nt_gf0,
                                              epsilon_d=epsilon_d,
                                              muL=muL, muR=muR)


S = SolverCore(g0_lesser, g0_greater)

#The final lists
on_list, cn_last, on_last,sn_last = [], [], [],[]

#The lists for the relative errors
sn_errors_last,pn_errors_last = [],[]
on_relative_errors = []



if mpi.rank==0:
    name_file = 'final_results.h5'
    R = HDFArchive(name_file, 'w') # A file to store the results

#Setting the verbosity to 3 only for the master
verb = 0
if mpi.rank ==0:
    verb=3

for order in range(0, max_order):

    (pn, sn),(pn_error,sn_error) = S.solve(U=U_qmc,
                 max_perturbation_order=order,
                 min_perturbation_order=0,
                 p_dbl=p_dbl,
                 tmax=tmax,
                 alpha=alpha,
                 verbosity=verb,
                 n_cycles=n_cycles,
                 n_warmup_cycles=n_warmup_cycles,
                 random_seed=random_seed,
                 length_cycle=length_cycle)

    inv_U_qmc = 1. / U_qmc
    cn_over_Zqmc = np.array([x * inv_U_qmc ** n for n, x in enumerate(pn)])
    fact = cn_over_Zqmc[-2] / c_norm if order > 0 else 1
    cn = cn_over_Zqmc / fact
    on = cn * sn
    on_list.append(on)

    #Saving the values
    cn_last.append(cn[-1])
    on_last.append(on[-1])
    sn_last.append(sn[-1])

    #Saving the errors
    pn_errors_last.append(pn_error[-1]/pn[-1]) #This is the same one as the cn normalized deviation.
    sn_errors_last.append(sn_error[-1]/np.abs(sn[-1])) #We take the normalized deviation
    on_relative_errors.append(pn_errors_last[-1] + sn_errors_last[-1])

    c_norm = cn[-1]  # for next iter

    if mpi.rank == 0:
        print "----------- order = ", order, "---------------"
        print "cn = ",cn
        print " pn = ", pn
        print "sn = ", sn
        print "sn normalized deviation = ", sn_errors_last
        print "pn/z normalized deviation = ", pn_errors_last
        print "on = ", on
        print "on_last = ", on_last


        #We save the results
        if order==0:
            R['on_values'] = np.array([on_last[0]])
            R['on_errors'] = np.array([on_relative_errors[0]])
            R['cn_values'] = np.array([cn_last[0]])
            R['cn_errors'] = np.array([pn_errors_last[0]])
            R['sn_values'] = np.array([sn_last[0]])
            R['sn_errors'] = np.array([sn_errors_last[0]])
        else:
            R['on_values'] = np.append(R['on_values'],on_last[-1])
            R['on_errors'] = np.append(R['on_errors'],on_relative_errors[-1])
            R['cn_values'] = np.append(R['cn_values'],cn_last[-1])
            R['cn_errors'] = np.append(R['cn_errors'],pn_errors_last[-1])
            R['sn_values'] = np.append(R['sn_values'],sn_last[-1])
            R['sn_errors'] = np.append(R['sn_errors'],sn_errors_last[-1])




if mpi.rank == 0:
    print "--------- on ---------"
    print on_last
    print "-------Relative error----"
    print on_relative_errors
    del R



