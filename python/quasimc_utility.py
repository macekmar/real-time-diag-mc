import os
import re

import numpy as np
from mpi4py import MPI
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator

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

    # Process random seed
    if params_py['random_seed'] is False:
        seed = generator.default_seed
        if world.rank == 0:
            print("Setting the seed to the default value: %d of the generator: %s" % (generator.default_seed, str(generator)))
    else:
        seed = params_py['random_seed']

    # Check if the results file already exist
    if params_py['filemode'] is 'w' and os.path.isfile(params_py['filename']):
        raise IOError("File %s already exists!" % params_py['filename'])

    return t_min, model, generator, nb_bins_sum, random_shift, seed, overwrite, params_py, params_cpp


def fix_cpp_parameters(p_cpp, p_py):
    if 'max_perturbation_order' not in p_cpp:
        p_cpp['max_perturbation_order'] = max(p_py['order'])
    if 'min_perturbation_order' not in p_cpp:
        p_cpp['min_perturbation_order'] = 0
    if 'U' not in p_cpp:
        p_cpp['U'] = [1.0]*p_cpp['max_perturbation_order']
    if 'sampling_model_intervals' not in p_cpp:
        p_cpp['sampling_model_intervals'] = [[]]
    if 'sampling_model_coeff' not in p_cpp:
        p_cpp['sampling_model_coeff'] = [[[]]]


def _calculate_inv_cdf(fun, t_min, t_max=0, Nt=1001):
    """Calculates inverse CDF for a nonnormalized function."""
    u_lin = np.linspace(t_min, t_max, Nt)
    fun_val = np.abs(fun(u_lin[:, np.newaxis])) # Newaxis is necessary for the get function
    cdf = cumtrapz(fun_val, u_lin, initial=0)
    integral = cdf[-1]
    cdf = cdf/integral
    return integral, PchipInterpolator(cdf, u_lin)

def calculate_inv_cdfs(funs, t_min, t_max=0, Nt=1001):
    """Calculates inverse CDF for nonnormalized functions."""
    integral = [None for i in range(len(funs))]
    inv_cdf = [None for i in range(len(funs))]
    for i, f in enumerate(funs):
        integral[i], inv_cdf[i] = _calculate_inv_cdf(f, t_min=t_min, t_max=t_max, Nt=Nt)

    return integral, inv_cdf
