import os
from datetime import datetime

import numpy as np

from ctint_keldysh.quasimc_results import *
from ctint_keldysh.solver import _add_params_to_results
from pytriqs.archive import HDFArchive

# Test if data is saved properly?

orders = [2,3,5]
N_vec = [0, 100, 200, 300]
model_ints = np.random.rand(5)
params_cpp = {'nb_orbitals': 2, 'nb_bins': 10000}
params_py = {'frequency_range': False, 'nb_bins_sum': 1, 'filemode': 'a', 'filename': 'TestSaving.hdf', 'run_name': "Run"}

start_time = datetime.now()

# Code copied from quasimc.py
results_to_save = create_empty_results(orders, N_vec, params_py, params_cpp)
_add_params_to_results(results_to_save, dict(params_cpp, **params_py))
metadata = {}
metadata['total_duration'] = 0.0
metadata['order_duration'] = np.empty(len(orders), dtype=np.float)
metadata['orders'] = orders
metadata['model_integrals'] = model_ints
results_to_save['metadata'] = metadata
# TODO: generator, seed, random shift

params_py['run_name'] = save_empty_results(results_to_save, params_py['filename'], params_py['run_name'], params_py['filemode'])
filesize_org = os.path.getsize(params_py['filename'])
params_py['run_name'] = save_empty_results(results_to_save, params_py['filename'], params_py['run_name'], params_py['filemode'])
filesize = os.path.getsize(params_py['filename'])

assert ((filesize_org - filesize) == 0), "Different file size after resave in filemode %s" % params_py['filemode']

# Prepare results
results = {"results_final": {}, "results_inter": {}}
for res in ["results_final", "results_inter"]:
    for key in results_to_save[res].keys():
        typ = results_to_save[res][key].dtype
        shp = results_to_save[res][key].shape
        if typ == np.int:
            results[res][key] = np.random.randint(0,1000, shp)
        if typ == np.float:
            results[res][key] = np.random.rand(*shp)
        if typ == np.complex:
            results[res][key] = np.random.rand(*shp) + 1j*np.random.rand(*shp)
        # Cppy to results_to save
        results_to_save[res][key] = results[res][key].copy()
# Save results_to_save
params_py['run_name'] = save_empty_results(results_to_save, params_py['filename'], params_py['run_name'], params_py['filemode'])
# time.sleep(30)

# Now we can modify results_to_save in a loop
for io, order in enumerate(orders):
    for iN in range(len(N_vec) - 1):
        # Generate chunk_results
        chunk_results = {}
        shp = (max(orders), params_cpp['nb_bins'], 2, params_cpp['nb_orbitals'])
        chunk_results['bin_times'] = np.linspace(-1,0, params_cpp['nb_bins'])
        chunk_results['kernels'] = np.random.rand(*shp) + 1j*np.random.rand(*shp)
        chunk_results['nb_kernels'] = np.random.randint(0,1000, shp)
        chunk_results['dirac_times'] = np.array([1.0])
        shp = (max(orders), 1, 2, params_cpp['nb_orbitals'])
        chunk_results['kernel_diracs'] = np.random.rand(*shp) + 1j*np.random.rand(*shp)
        chunk_results['N_calculated'] = N_vec[iN+1]
        chunk_results['N_generated'] = 2*N_vec[iN+1]

        metadata['total_duration'] = (datetime.now() - start_time).total_seconds()
        metadata['order_duration'] = (datetime.now() - start_time).total_seconds()

        # Update saved results
        # Number of bytes written takes some time -- I don't know if pauses are really necessary -- and is not precise
        # time.sleep(30)
        # disk_io_bytes_start = psutil.disk_io_counters().write_bytes
        
        update_results(chunk_results, metadata, io, order, iN, 1, params_py)
        # time.sleep(10)
        # disk_io_bytes_end = psutil.disk_io_counters().write_bytes
        # disk_io_diff = disk_io_bytes_end - disk_io_bytes_start
        
        filesize = os.path.getsize(params_py['filename'])
        assert ((filesize_org - filesize) == 0), "Different file size after resave in filemode %s" % params_py['filemode']
        # print io, iN, "FS: %3.2f MB, Write: %3.2f MB" % (filesize/1024.0**2, disk_io_diff/1024.0**2)


        # Update results for comparison
        for key in ['kernels', 'nb_kernels', 'kernel_diracs']:
            results['results_final'][key][io, ...] = chunk_results[key][order-1, ...]
        for key in ['N_generated', 'N_calculated']:
            results['results_final'][key][io] = chunk_results[key]
        for key in ['bin_times', 'dirac_times']:
            results['results_final'][key][...] = chunk_results[key] # [...] is necessary!

        # Saved results and results for comparison should match
        with HDFArchive(params_py['filename'], 'r') as f:
            run = f["Run"]
            for key in run['results_final'].keys():
                assert np.allclose(results['results_final'][key], run['results_final'][key])

            for key in run['results_final'].keys():
                if key in ['kernels', 'nb_kernels', 'kernel_diracs']:
                    # Can only compare the most recent one
                    assert np.allclose(chunk_results[key][order-1], run['results_inter'][key][io,..., iN])
                elif key in ['N_generated', 'N_calculated']:
                    assert chunk_results[key] == run['results_inter'][key][io, iN]
                else:
                    assert np.allclose(chunk_results[key], run['results_inter'][key]) 


print "SUCCESS!"
os.remove("TestSaving.hdf")