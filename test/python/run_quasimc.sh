# This scipt tests if results for different number of processes but for the
# same  points are the same

rm Test*.hdf5

mpiexec -np 1 python quasimc.py 1200
mpiexec -np 2 python quasimc.py 600
mpiexec -np 3 python quasimc.py 400
mpiexec -np 4 python quasimc.py 300
mpiexec -np 5 python quasimc.py 240
mpiexec -np 10 python quasimc.py 120

python process_quasimc.py

rm Test*.hdf5