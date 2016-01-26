from pytriqs.gf.local import *
from pytriqs.utility import mpi
from pytriqs.applications.impurity_solvers.ctint_new import CtintSolver

S = CtintSolver()
c_norm = 0
qn_list, cn_list =[], []

def fact(i) : 
    if i<=0 : return 1
    return i*fact(i-1)

for Nmax in range(1,5) : 
#for Nmax in range(8,10) : 

  S.solve(U = 1, max_perturbation_order = Nmax, beta = 100.0, tmax = 1.0, Gamma = 1.0, alpha = 0.0, 
        n_cycles = 30000000, n_warmup_cycles = 100000, length_cycle=1)

  cn,sn  = S.CnSn[0,0:Nmax+1], S.CnSn[1,0:Nmax+1]
  #print cn
  cn = cn/ (cn[1])
  #print cn
  print "cn = ", [cn[i] * fact(i) for i in range(len(cn))]

