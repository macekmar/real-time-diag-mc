from pytriqs.gf.local import *
from pytriqs.utility import mpi
from pytriqs.applications.impurity_solvers.ctint_new import CtintSolver

S = CtintSolver(beta = 100.0, Gamma = 0.25, epsilon_d = 0.0)
c_norm = 0
qn_list, cn_list =[], []

for Nmax in range(1,4) : 

  S.solve(U = 1, max_perturbation_order = Nmax, tmax = 10.0, alpha = 0.0,
          n_cycles = 1000000, n_warmup_cycles = 10000, length_cycle=10)

  cn,sn  = S.CnSn[0,0:Nmax+1], S.CnSn[1,0:Nmax+1]
  if Nmax == 1 : 
      c_norm = abs(S.c0) # first computation
      cn_last = [c_norm]
      qn_last = [sn[0]]
  cn = cn/ (cn[-2]/c_norm)
  c_norm = cn[-1]
  qn_list.append(cn * sn)
  cn_list.append(cn)
  cn_last.append(cn[-1])
  qn_last.append(cn[-1]*sn[-1])

  print "cn = ", cn
  print "sn = ", sn
  print "qns = ", qn_list

for q in  cn_list: 
    print q

for q in  qn_list: 
    print q

print "cn_last", cn_last
print "qn_last", qn_last

def nd(U) : 
    u,r  = 1,0
    for q in qn_last : 
       r = r + q*u
       u= u*U
    return r


