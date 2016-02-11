import os as os
from sys import maxint
import random as random

Gamma = 1.0
beta = 200.
GF_type = 1 #Werner
is_current = True
length_cycle = 3
n_cycles = 100000
n_warmup_cycles = 100
Nmax = 2
U_qmc = 2.
p_dbl = 1.
if_ph = True

ens_alpha = [0.5]
ens_ed =    [0.]
ens_V =     [0.2 * Gamma, 0.4 * Gamma]
ens_tmax =  [4., 6.]
ens_seed=[1]
verbosity = 3
if_adjust_U = True

                
for epsilon_d in ens_ed:
  for alpha in ens_alpha: 
    for V in ens_V:
      muL = V/2.
      muR = -V/2.
      for tmax in ens_tmax:
        for random_seed in ens_seed:
            # we characterize the job by a random number
            num=random.randint(0,maxint)
            print num, ', ed=',epsilon_d,', alpha=', alpha, ', V=', V, ', tmax=', tmax, ', seed=', random_seed
            filename0 = 'job_'+str(num)+'.py' # python file of the job
            filename = 'job_'+str(num)+'.sh'         # slurm file for the job
            
            # python file is written here
            f0=open(filename0, 'w')
            f0.write('from qmc_keldysh import *'+'\n')
            f0.write('p = SimpleNamespace(alpha = '+str(alpha)+',\n')
            f0.write('  epsilon_d='+str(epsilon_d)+',\n')
            f0.write('  Gamma='+str(Gamma)+',\n')
            f0.write('  tmax='+str(tmax)+',\n')
            f0.write('  muL='+str(muL)+',\n')
            f0.write('  muR='+str(muR)+',\n')
            f0.write('  beta='+str(beta)+',\n')
            f0.write('  is_current ='+str( is_current)+',\n')
            f0.write('  GF_type = '+str(GF_type)+')\n')
            f0.write('p_qmc = SimpleNamespace(U_qmc = '+str(U_qmc)+',\n')
            f0.write('  n_cycles = '+str(n_cycles)+',\n')
            f0.write('  n_warmup_cycles ='+str( n_warmup_cycles)+',\n') 
            f0.write('  length_cycle = '+str(length_cycle)+',\n') 
            f0.write('  LastOrder='+str(Nmax)+',\n')
            f0.write('  p_dbl='+str(p_dbl)+',\n')
            f0.write('  random_seed='+str(random_seed)+')\n')
            f0.write('res = QMC(p,p_qmc,if_adjust_U='+str(if_adjust_U)+', verbosity='+str(verbosity)+', if_ph='+str(if_ph)+')\n')
            f0.write("save(p,p_qmc,res, 'resultats_"+str(num)+".h5')\n")
            f0.close()
            
            f = open(filename, 'w')
            f.write('#!/bin/bash\n')
            f.write('#SBATCH --job-name=ct_int_keldysh\n')
            f.write('#SBATCH --output=job_'+str(num)+'.out\n')
            f.write('#SBATCH --error=job_'+str(num)+'.err\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=12:00:00\n')
            f.write('#SBATCH --mem-per-cpu=100\n')
            f.write('#SBATCH --partition=normal\n')
            f.write('srun nice -n 19 python '+filename0+'\n')
            f.close()
            
            # we run the job
            os.system('sbatch '+filename)