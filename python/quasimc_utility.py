import os
import re

import numpy as np
from mpi4py import MPI
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator

from solver import extract_and_check_params


def process_parameters(params, default_py, default_cpp):
    """Extract parameters into different variables and do some checks."""
    world = MPI.COMM_WORLD
    ### Extract parameters 
    t_g0_max = params['g0_lesser'].mesh.t_max # extract_... deletse g0, doesn't keep the data
    params_py = extract_and_check_params(params, default_py)
    params_cpp = extract_and_check_params(params, default_cpp)
    t_start = params_cpp['interaction_start']
    
    generator = params_py['num_gen']
    model = params_py['model'][:]
    # Remove 'model' and 'num_gen', which cannot be saved, g0_lesser is taken care of in _add_params_to_results
    del params_py['model']
    del params_py['num_gen']

    ### Do some checks
    # Method
    # if params_cpp['method'] == 0:
    #     raise Exception("Method 0 is not implemented yet.")

    # We do not want to reduce the binning before the Fourier transform
    if params_py['frequency_range'] is False:
        nb_bins_sum = params_py['nb_bins_sum']
    else:
        nb_bins_sum = 1

    intp_pts = None
    if params_py['interpolation_points'] is False:
        intp_pts = np.linspace(0.0, t_start, 1001)
    elif isinstance(params_py['interpolation_points'], int):
        intp_pts = np.linspace(0.0, t_start, params_py['interpolation_points'])
    elif isinstance(params_py['interpolation_points'], tuple):
        t_min, t_max, nb_points = params_py['interpolation_points']
        intp_pts = np.linspace(t_min, t_max, nb_points)
    elif isinstance(params_py['interpolation_points'], list) or isinstance(params_py['interpolation_points'], np.ndarray):
        intp_pts = params_py['interpolation_points']
    if intp_pts is None:
        raise Exception("Wrong format for parameter 'interpolation points'")

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
                print "Changing run_name to %s" % name
    else:
        random_shift = np.zeros((max(params_py['order']),))
        overwrite = True

    # Process random seed
    if params_py['num_gen_seed'] is False:
        seed = generator.default_seed
        if world.rank == 0:
            print "Setting the seed to the default value: %d of the generator: %s" % (generator.default_seed, str(generator))
    else:
        seed = params_py['num_gen_seed']

    # Procees num_gen_kwargs
    num_gen_kwargs = params_py['num_gen_kwargs']

    # Check if the results file already exist
    if params_py['filemode'] is 'w' and os.path.isfile(params_py['filename']):
        raise IOError("File %s already exists!" % params_py['filename'])

    # Get save_period
    save_period = None
    period = params_py['save_period']
    if isinstance(period, int):
        save_period = [period for o in params_py['order']]
    if isinstance(period, list):
        if len(period) == len(params_py['order']) and all(isinstance(p, int) for p in period):
            save_period = period
    if save_period is None:
        raise ValueError("Parameter save_period is not an int or a list of ints.")

    return t_start, t_g0_max, model, intp_pts, generator, nb_bins_sum, random_shift, seed, num_gen_kwargs, save_period, overwrite, params_py, params_cpp


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


def _calculate_inv_cdf(fun, pts):
    """Calculates inverse CDF for a nonnormalized function."""
    fun_val = np.abs(fun(pts[:, np.newaxis])) # Newaxis is necessary for the get function
    cdf = cumtrapz(fun_val, pts, initial=0)
    integral = cdf[-1]
    cdf = cdf/integral
    return integral, PchipInterpolator(cdf, pts)

def calculate_inv_cdfs(funs, points):
    """Calculates inverse CDF for nonnormalized functions."""
    if isinstance(points, tuple):
        t_min, t_max, Nt = points
        pts = np.linspace(t_min, t_max, Nt)
    elif isinstance(points, list) or isinstance(points, np.ndarray):
        pts = points
    else:
        raise Exception("You either have to provide a tuple (t_min, t_max, Nt) for np.linspace for interpolation points or provide points themselves.")
    integral = [None for i in range(len(funs))]
    inv_cdf = [None for i in range(len(funs))]
    for i, f in enumerate(funs):
        integral[i], inv_cdf[i] = _calculate_inv_cdf(f, pts)

    return integral, inv_cdf