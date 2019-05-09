import os
from solver import extract_and_check_params


def process_parameters(params, default_py, default_cpp):
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

    # Check if the results file already exist
    if params_py['filemode'] is 'w' and os.path.isfile(params_py['filename']):
        raise IOError("File %s already exists!" % params_py['filename'])

    return t_min, model, generator, nb_bins_sum, params_py, params_cpp
