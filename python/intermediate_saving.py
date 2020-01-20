from mpi4py import MPI
import numpy as np
import h5py
from pytriqs.archive import HDFArchive
from results import _compute_cn_v, _compute_cn, _safe_divide
from copy import deepcopy

def create_empty_results(orders, N_vec, params_py, params_cpp):
    lo = len(orders)
    lN = len(N_vec)
    lc = MPI.COMM_WORLD.size

    results = {'results': {}}
    if params_cpp['method'] == 0:
        results['results']['sn'] = np.zeros((lN, lo-1, lo, lc), dtype=np.complex)
        results['results']['pn'] = np.zeros((lN, lo-1, lo, lc), dtype=np.int)
        results['results']['U'] = np.zeros((lo-1, lo-1), dtype=np.float)

        results['metadata'] = {}
        results['metadata']['duration'] = np.zeros(1)
        results['metadata']['durations'] = np.zeros((lN, lo-1), dtype=float)
        results['metadata']['nb_measures'] = np.zeros((lN, lo-1), dtype=np.int)
        results['metadata']['nb_proc'] = MPI.COMM_WORLD.size
    else:
        raise Exception("Only method 0 is implemented")

    return results

def save_empty_results(results, filename, run_name, overwrite=True, filemode="w"):
    """
    Saves the `results` dictionnary in `filename` as an hdf5 archive under the key `run_name`.

    If this key already exists, it is overwritten (default) or a new one is
    generated (if `overwrite` is False).

    Three groups are created under this key:
    > metadata
    > parameters
    > results
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
        group_names = ['metadata', 'parameters', 'results']
        for name in group_names:
            run.create_group(name)
            group = run[name]
            for key in results[name]:
                # num_gen_kwargs is a dictionary
                if name == 'parameters' and key == 'num_gen_kwargs':
                    group.create_group(key)
                    for subkey in results[name][key]:
                        group[key][subkey] = results[name][key][subkey]
                else:
                    group[key] = results[name][key]

    print 'Saved in {0}\n'.format(filename)
    return run_name

def update_results(chunk_results, order, iN, params_py, params_cpp):
    assert MPI.COMM_WORLD.rank == 0

    with h5py.File(params_py["filename"], "a") as ar:  #HDF does't have r+ mode    

        run = ar[params_py["run_name"]]
        
        if params_cpp['method'] == 0:
            run['results']['sn'][iN, order-1, :order+1,:] = chunk_results['results_part']['sn']
            run['results']['pn'][iN, order-1, :order+1,:] = chunk_results['results_part']['pn']
            run['results']['U'][order-1,:order] = chunk_results['results_part']['U']

            run['metadata']['duration'][0] = run['metadata']['duration'][0] +  chunk_results['metadata']['duration']
            run['metadata']['durations'][iN, order-1] = chunk_results['metadata']['duration']
            run['metadata']['nb_measures'][iN, order-1] = chunk_results['metadata']['nb_measures']
        else: 
            raise Exception("Only method 0 is implemented")


def load_intermediate_results(filename):
    """
    Gathers data from staircases from a run with intermediate savings. This 
    produces optimal sn, cn for each N, which are equal to independent runs
    with the same N (nb_cycles).

    results["results_x"][yn] have an additional axis for N (the first axis).
    """

    results = {"results_all": {}, "results_part": {}}
    with HDFArchive(filename, "r") as f:
        run = f["Run"]
        U = run["results"]["U"]
        pn = run["results"]["pn"]
        sn = run["results"]["sn"]

        results["results_part"]["pn"] = pn
        pn_temp = np.sum(pn, axis=1)
        results["results_all"]["pn"] = np.sum(pn,axis=3)
        pn_summed= np.sum(pn_temp, axis=2)

        results["results_part"]["sn"] = _safe_divide(np.sum(sn*pn,axis=1), pn_temp)
        results["results_all"]["sn"] = _safe_divide(np.sum(results["results_part"]["sn"]*pn_temp, axis=2), pn_summed)

        results["results_part"]["cn"] = np.array([_compute_cn_v(temp, U) for temp in results["results_part"]["pn"]])
        temp = np.rollaxis(results["results_all"]["pn"], 0, 3) # Sum over all partitions, then treat N-index as a partition for `_compute_cn_v`
        results["results_all"]["cn"] = _compute_cn_v(temp, U).T

        results["metadata"] = dict(run["metadata"])
        results["parameters"] = dict(run["parameters"])
    return results