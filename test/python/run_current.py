from pytriqs.gf.local import *
from pytriqs.utility import mpi
from pytriqs.applications.impurity_solvers.ctint_new import CtintSolver
import math

alpha = 0# 0.5
U = 1 #5
U_qmc = 0.5
V = 0.8 #0.4#0.625 
epsilon_d = 0 - U * alpha
tmax = 100.0

S = CtintSolver(beta = 200.0, Gamma = 0.25,#*0.25, 
                epsilon_d = epsilon_d + U*alpha, 
                muL = V, muR = 0)

is_current = True #False
if is_current:
    c_norm = abs(S.I_L) # current
    print "current 0", S.I_L
else :
    c_norm = abs(S.c0) # density

qn_list, cn_list, qn_last =[[c_norm]], [[c_norm]], [c_norm]

for Nmax in range(1,2) : 

  S.solve(U = U_qmc,  
          is_current = is_current, 
          max_perturbation_order = Nmax, 
          tmax = tmax, 
          alpha = alpha,
          verbosity= 3, 
          n_cycles = 1000000, n_warmup_cycles = 10000, length_cycle=10)

  cn, sn  = S.CnSn[0,0:Nmax+1], S.CnSn[1,0:Nmax+1]
  f = float(U)/U_qmc
  cn_brut = cn
  cn = [ x * f ** n for n, x in enumerate(cn)]
  cn = cn/(cn[-2]/c_norm)
  qn = cn * sn
  qn_list.append(qn)
  cn_list.append(cn)
  qn_last.append(qn[-1])
  c_norm = cn[-1] # for next iter

  if mpi.rank == 0 : 
    print "cn = ", cn
    print "cn_brut = ", cn_brut
    print "sn = ", sn
    print "qn = ", qn
    print " qn_last = ", qn_last

if 0 : 
    print "--------- cn ---------"
    for q in  cn_list: 
      print q

if mpi.rank == 0 : 
  print "--------- qn ---------"
  for q in qn_list: 
     print "series = ", q
  print " qn_last = ", qn_last
  for n in range(len(qn_last)) : 
      print "partial sum", sum(qn_last[0:n+1])
  print "sum = ", sum(qn_last)
  

