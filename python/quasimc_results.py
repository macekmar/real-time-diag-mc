import numpy as np
from mpi4py import MPI
from pytriqs.archive import HDFArchive
from fourier_transform import fourier_transform, _next_regular
from solver import reduce_binning, _next_name_gen
import h5py

def extract_results(solver):
    res = {}
    for key in ['bin_times', 'dirac_times', 'kernels', 'nb_kernels', 'kernel_diracs']:
        res[key] =  np.array(getattr(solver, key))
    #res['U'] = np.array(solver.U)[np.newaxis, :]
    return res

def create_empty_results(orders, N_vec, params_py, params_cpp):
    results = {'results_final': {}, 'results_inter': {}}
    lo = len(orders)
    lN = len(N_vec) - 1 # The first N is 0, we skip it
    nb_orb = params_cpp['nb_orbitals']
    nb_bins = params_cpp['nb_bins']

    # results_all
    results['results_final']['bin_times'] = np.zeros((nb_bins, ), dtype=np.float)
    results['results_final']['dirac_times'] = np.zeros((1,), dtype=np.float)
    results['results_final']['kernel_diracs'] = np.zeros((lo, 1, 2, nb_orb), dtype=np.complex)
    results['results_final']['kernels'] = np.zeros((lo, nb_bins, 2, nb_orb), dtype=np.complex)
    results['results_final']['nb_kernels'] = np.zeros((lo, nb_bins, 2, nb_orb), dtype=np.int)
    results['results_final']['N_generated'] = np.zeros((lo, ), dtype=np.int)
    results['results_final']['N_calculated'] = np.zeros((lo, ), dtype=np.int)

    # resutls_part
    results['results_inter']['dirac_times'] = np.zeros((1,lN), dtype=np.float)
    results['results_inter']['kernel_diracs'] = np.zeros((lo, 1, 2, nb_orb, lN), dtype=np.complex)
    
    if params_py['frequency_range'] is False:
        # Reduced binning
        nb_bins_reduced = nb_bins - nb_bins % params_py['nb_bins_sum']
        results['results_inter']['bin_times'] = np.zeros((nb_bins_reduced, ), dtype=np.float)
        results['results_inter']['kernels'] = np.zeros((lo, nb_bins_reduced, 2, nb_orb, lN), dtype=np.complex)
        results['results_inter']['nb_kernels'] = np.zeros((lo, nb_bins_reduced, 2, nb_orb, lN), dtype=np.int)
    else:
        nb_omega = _next_regular(params_py['frequency_range'][-1])
        results['results_inter']['omega'] = np.zeros((nb_omega, ), dtype=np.float)
        results['results_inter']['kernels'] = np.zeros((lo, nb_omega, 2, nb_orb, lN), dtype=np.complex)

    results['results_inter']['N_generated'] = np.zeros((lo, lN), dtype=np.int)
    results['results_inter']['N_calculated'] = np.zeros((lo, lN), dtype=np.int)

    return results

def save_empty_results(results, filename, run_name, overwrite=True, filemode='w'):
    """
    Saves the `results` dictionnary in `filename` as an hdf5 archive under the key `run_name`.

    If this key already exists, it is overwritten (default) or a new one is
    generated (if `overwrite` is False).

    Four groups are created under this key:
    > metadata
    > parameters
    > results_all
    > results_part
    They are filled with content found in `results` under the same names.
    """

    assert MPI.COMM_WORLD.rank == 0 # avoid multiple processes to save

    if filemode == "w" and overwrite is False:
        raise Exception("Filemode 'w' clashes with overwrite=False.")

    ### create archive and fill it with leading elements and metadata
    with h5py.File(filename, filemode) as ar:  # h5py so that it is compatible with update_...() below
        ### run name
        if run_name in ar:
            if overwrite:
                del ar[run_name]
            else:
                ### Change run_name by appending a (or increasing the) trailling number
                nng = _next_name_gen(run_name)
                run_name = nng.next()
                while run_name in ar:
                    run_name = nng.next()

        ### new empty group
        ar.create_group(run_name)
        run = ar[run_name]

        ### fill it with results
        group_names = ['metadata', 'parameters', 'results_final', 'results_inter']
        for name in group_names:
            run.create_group(name)
            group = run[name]
            for key in results[name]:
                group[key] = results[name][key]

    print 'Saved in {0}\n'.format(filename)
    return run_name



def update_results(chunk_results, metadata, io, order, iN, nb_bins_sum, params_py):
    assert MPI.COMM_WORLD.rank == 0

    with h5py.File(params_py["filename"], "a") as ar:  #HDF does't have r+ mode
        run = ar[params_py["run_name"]]
        # update results_all
        for key in ['kernels', 'nb_kernels', 'kernel_diracs']:
            run['results_final'][key][io, ...] = chunk_results[key][order-1, ...]
        for key in ['N_generated', 'N_calculated']:
            run['results_final'][key][io] = chunk_results[key]
        for key in ['bin_times', 'dirac_times']:
            run['results_final'][key][...] = chunk_results[key] # [...] is necessary!

        # update results_intermediate
        res_part = dict(chunk_results)
        res_part['bin_times'] = reduce_binning(res_part['bin_times'], nb_bins_sum) / float(nb_bins_sum)
        res_part['kernels'] = reduce_binning(res_part['kernels'], nb_bins_sum, axis=1) / float(nb_bins_sum)
        res_part['nb_kernels'] = reduce_binning(res_part['nb_kernels'], nb_bins_sum, axis=1) # no normalization !

        # Do Fourier
        if params_py['frequency_range'] is not False:   
            w_window = params_py['frequency_range'][0:2]
            nb_w = params_py['frequency_range'][2]
            w, kernels_w = fourier_transform(res_part['bin_times'], res_part['kernels'],
                                            w_window, nb_w, axis=1)
            res_part['omega'] = w
            res_part['kernels'] = kernels_w

            del res_part['bin_times']
            del res_part['nb_kernels']


        for key in run['results_final']:
            if key in ['kernels', 'nb_kernels', 'kernel_diracs']:
                if key in res_part: # after FFT, nb_kernels are missing
                    run['results_inter'][key][io, ..., iN] = res_part[key][order - 1, ...] 
            elif key in ['N_generated', 'N_calculated']:
                run['results_inter'][key][io, iN] = res_part[key]        
            else:
                run['results_inter'][key][...] = res_part[key]  

        run["metadata"]['total_duration'][...] = metadata['total_duration']
        run["metadata"]['order_duration'][io] = metadata['order_duration']
        
    #print 'Updated in {0}'.format(params_py["filename"])          



if __name__ == '__main__':
    print 'Start tests'

    from mpi4py import MPI
    from datetime import datetime
    from solver import _add_params_to_results
    import os
    # import psutil
    # import time
    from pytriqs.archive import HDFArchive

    world = MPI.COMM_WORLD

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
    metadata['nb_proc'] = world.size
    metadata['orders'] = orders
    metadata['model_integrals'] = model_ints
    results_to_save['metadata'] = metadata
    # TODO: generator, seed, random shift
    if world.rank == 0:
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

    