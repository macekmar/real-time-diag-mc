import numpy as np
from pytriqs.archive import HDFArchive

kernels = {}
nb_kernels = {}

world_sizes = [1,2,3,4,5,10]

for ws in world_sizes:
    with HDFArchive("Test" + str(ws) + ".hdf5", "r") as f:
        kernels[ws] = f["Run"]["results_final"]["kernels"]
        nb_kernels[ws] = f["Run"]["results_final"]["nb_kernels"]

for ws_1 in world_sizes: 
    for ws_2 in world_sizes:     
        assert np.allclose(kernels[ws_1], kernels[ws_2])
        assert np.allclose(nb_kernels[ws_1], nb_kernels[ws_2])
