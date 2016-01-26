from pytriqs.gf.local import *
from pytriqs.utility import mpi
from pytriqs.applications.impurity_solvers.ctint_new import CtintSolver
import numpy as np

for V in np.arange(0,3,0.1) : 
  S = CtintSolver(beta = 100.0, Gamma = 0.25, epsilon_d = 0.0, muL =V)
  print V, S.I_L

