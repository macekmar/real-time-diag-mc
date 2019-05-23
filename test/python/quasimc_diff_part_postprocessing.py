import numpy as np
from pytriqs.archive import HDFArchive

# Part of run_quasimc_diff_part.sh

kernels = {}
nb_kernels = {}
N_calculated = {}

world_sizes = [1,2,3,4,5,10]

for ws in world_sizes:
    with HDFArchive("Test" + str(ws) + ".hdf5", "r") as f:
        kernels[ws] = f["Run"]["results_inter"]["kernels"]
        nb_kernels[ws] = f["Run"]["results_inter"]["nb_kernels"]
        N_calculated[ws] = f["Run"]["results_inter"]["N_calculated"]
        orders = f["Run"]["metadata"]["orders"]

for ws_1 in world_sizes: 
    for ws_2 in world_sizes:     
        assert np.allclose(kernels[ws_1], kernels[ws_2])
        assert np.allclose(nb_kernels[ws_1], nb_kernels[ws_2])

for ws_1 in world_sizes:
    N_diff = [np.sum(nb_kernels[ws][0,:,0,0,i])/orders[0] - N_calculated[ws][0,i] == 0 for i in range(len(N_calculated[ws][0]))]
    assert  all(N_diff)

print("Finished!")