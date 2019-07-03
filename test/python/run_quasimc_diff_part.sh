#!/bin/bash
# This scipt tests if results for different number of processes but for the
# same  points are the same

rm Test*.hdf5

# You cannot change number of samples freely. it has to be divisible by 10
# for all partitionings (check quasimc_diff_part.py line 54)
# on the other hand it also has to be divisible by 2,3,4,5.
mpiexec -np 1 python quasimc_diff_part.py 4800
mpiexec -np 2 python quasimc_diff_part.py 2400
mpiexec -np 3 python quasimc_diff_part.py 1600
mpiexec -np 4 python quasimc_diff_part.py 1200
mpiexec -np 5 python quasimc_diff_part.py 960
mpiexec -np 10 python quasimc_diff_part.py 480

python quasimc_diff_part_postprocessing.py

rm Test*.hdf5