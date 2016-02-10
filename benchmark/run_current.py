from pytriqs.gf.local import *
from pytriqs.utility import mpi
from ctint_keldysh import *
import math

alpha =  0.#0.5
U = 1 #5
U_qmc = 0.5
V = 0.8
epsilon_d = 0
last_order = 3

g0_lesser,g0_greater = make_g0_semi_circular(beta = 200.0, Gamma = 0.25,
                                             tmax_gf0 = 10.0, Nt_gf0 = 10000,
                                             epsilon_d = epsilon_d + U*alpha,
                                             muL = V, muR = 0)

S = SolverCore(g0_lesser,g0_greater)

for Nmax in range(0, last_order) :

  pn, sn = S.solve(U = U_qmc,
          max_perturbation_order = Nmax,
          min_perturbation_order = 0,
          p_dbl = 0,
          tmax = tmax,
          alpha = alpha,
          verbosity= 0,
          n_cycles = 100000, n_warmup_cycles = 1000, length_cycle=10)

  f = 1./U_qmc
  cn_over_Zqmc = [ x * f ** n for n, x in enumerate(pn)]
  fact = cn_over_Zqmc[-2]/c_norm if Nmax > 0 else 1
  cn = cn_over_Zqmc/fact
  qn = cn * sn
  qn_list.append(qn)
  cn_list.append(cn)
  qn_last.append(qn[-1])
  c_norm = cn[-1] # for next iter

  if mpi.rank == 0 :
    print "----------- Nmax = ", Nmax, "---------------"
    print "cn = ", cn
    print "sn = ", sn
    print "qn = ", qn
    print "qn_last = ", qn_last

if mpi.rank == 0 :
  print "--------- qn ---------"
  print qn_list
