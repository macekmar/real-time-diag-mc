import os
import re

import numpy as np
from mpi4py import MPI

from solver import extract_and_check_params


def process_parameters(params, default_py, default_cpp):
    world = MPI.COMM_WORLD
    """Extract parameters into different variables and do some checks."""
    ### Extract parameters 
    params_py = extract_and_check_params(params, default_py)
    params_cpp = extract_and_check_params(params, default_cpp)
    t_min = -params_cpp['interaction_start']
    generator = params_py['num_gen']
    model = params_py['model'][:]
    # Remove 'model' and 'num_gen', which cannot be saved, g0_lesser is taken care of in _add_params_to_results
    del params_py['model']
    del params_py['num_gen']

    ### Do some checks
    # Method
    if params_cpp['method'] == 0:
        raise Exception("Method 0 is not implemented yet.")

    # Generator mode
    if not((params_py['num_gen_mode'] is 'simple') or (params_py['num_gen_mode'] is 'complex')):
        raise Exception("Number generator mode should be 'simple' or 'complex'.")

    # We do not want to reduce the binning before the Fourier transform
    if params_py['frequency_range'] is False:
        nb_bins_sum = params_py['nb_bins_sum']
    else:
        nb_bins_sum = 1

    # Check random shift
    if params_py['random_shift'] is not False:
        random_shift = params_py['random_shift']
        if type(random_shift) is not np.ndarray:
            raise Exception("Parameter random_shift has to be a numpy array.")
        if len(random_shift.shape) > 1:
            raise Exception("Parameter random_shift has to be 1D numpy array")
        if random_shift.shape[0] < max(params_py['order']):
            raise Exception("Parameter random_shift has to be at least as large as the maximum order.")
        if params_py['filemode'] is 'w':
            raise IOError("If using random_shift, filemode has to be 'a'.")
        overwrite = False

        regex = re.compile(r'_[0-9]+$')
        name = params_py['run_name']
        match = regex.search(name)
        if match is None:
            name = name + '_1'
            params_py['run_name'] = name
            if world.rank == 0:
                print("Changing run_name to %s" % name)
    else:
        random_shift = np.zeros((max(params_py['order']),))
        overwrite = True

    # Check if the results file already exist
    if params_py['filemode'] is 'w' and os.path.isfile(params_py['filename']):
        raise IOError("File %s already exists!" % params_py['filename'])

    return t_min, model, generator, nb_bins_sum, random_shift, overwrite, params_py, params_cpp
