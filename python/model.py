import sobol
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator
from mpi4py import MPI

def _generate_sobol(dim, N, skip=1):                                           
    i = 1              
    nums = []                           
    for n in sobol.sobol_gen(dim, skip):
        i += 1         
        nums.append(n)
        if i > N:
            break
    return np.array(nums)

def get(solver, u, do_measure=False):
    """ Returns weights in points u."""
    return np.array([
        solver.evaluate_qmc_weight([(0, 0, c) for c in coors], do_measure) for coors in u
    ])

def model(funs, u):
    """Evaluates the spring model: f1(u1-u2)f2(u2-u3)..fn(un), u is ordered

    Times are first ordered as:
    \[
        -\infty < u_1 < u_2 < \dots < u_n < 0
    \]
    In general functions $f_i$ are different and stored in `self.sm_funs`
    as `[f_1 f_2 ... f_n]`.
    The spring model then returns
    \[
        f_1(u_1-u_2) f_2(u_2-u_3) \dots f_{n-1}(u_{n_1} - u_n) f_n(u_n)
    \]
    The last term can be imagined as f_n(u_n - u_{n+1}), 
    where u_{n+1} = 0. """

    order = u.shape[1]
    t = np.sort(u, axis=1)
    vals = 1.0
    for i in range(order-1):
        arg = (t[:,i:i+1] - t[:,i+1:i+2])
        vals *= funs[-(order-i)](arg)
        #vals *= self.get(arg)**(1+2*((i+1)%2)) # For symmetric case (params Corentin 2)
    vals *= funs[-1](t[:,-1:])
    return vals

def generate_u(inv_cdf, interaction_start, N_samples, dim, N_skip=0):
    """Generates u distributed by spring model.
    It generates N_samples v-variables. Some u-variables (true variables) 
    can lie outside of the domain [-interaction_start, 0]^order."""
    v = _generate_sobol(dim, N_samples, skip=N_skip+1)
    u = np.zeros_like(v)
    for i in range(-dim, 0):
        u[:,i] = inv_cdf[i](v[:,i])
    # Back to u from v
    for i in range(2, dim+1):
        u[:,-i] = u[:,-i] + u[:,-i+1]
    # Reject u out of t_min
    return u
   
def generate_u_complex(inv_cdf, interaction_start, N_samples, dim, N_skip=0):
    """Generates u distributed by spring model.
    It returns u points where at least N_samples are in the domain."""
    N_batch = N_samples
    itr = 0
    N = 0
    N_generated = 0
    while N < N_samples:
        # Inverse transform sampling
        v = _generate_sobol(dim, N_batch, skip=N_skip + N_generated + 1)
        u = np.zeros_like(v)
        for i in range(-dim, 0):
            u[:,i] = inv_cdf[i](v[:,i])
        # Back to u from v
        for i in range(2, dim+1):
            u[:,-i] = u[:,-i] + u[:,-i+1]
        # Reject u out of t_min
        inds = np.where(np.all(u > interaction_start, axis=1))[0]
        if N == 0:
            u_samples = u
        else:
            u_samples = np.vstack((u_samples, u))

        N += len(inds)
        N_generated += N_batch

        itr += 1
        if len(inds) != 0:
            N_batch = np.int(np.ceil( (N_samples - N)*max(1, 1.1*N_batch/float(len(inds)))))
        else:
            N_batch *= 2
    return u_samples #[:N_samples,:]


def _calculate_inv_cdf(fun, t_min, t_max=0, Nt=1001):
    """Calculates inverse CDF for a nonnormalized function."""
    u_lin = np.linspace(t_min, t_max, Nt)
    fun_val = np.abs(fun(u_lin[:, np.newaxis])) # Newaxis is necessary for the get function
    cdf = cumtrapz(fun_val, u_lin, initial=0)
    integral = cdf[-1]
    cdf = cdf/integral
    return integral, PchipInterpolator(cdf, u_lin)

def calculate_inv_cdfs(model, t_min, t_max=0, Nt=1001):
    
    world = MPI.COMM_WORLD
    # Split
    if world.rank == 0:
        print('Calculating inverse CDF')
        funs = model
        funs = np.array_split(funs, world.size)
    else:
        funs = None
    funs = world.scatter(funs, root=0)
    integral = [None for i in range(len(funs))]
    inv_cdf = [None for i in range(len(funs))]
    # Calculate
    for i, f in enumerate(funs):
        integral[i], inv_cdf[i] = _calculate_inv_cdf(f, t_min=t_min, t_max=t_max, Nt=Nt)
    # Gather
    integral = world.gather(integral, root=0)
    inv_cdf = world.gather(inv_cdf, root=0)
    if world.rank == 0:
        integral = [l for arr in integral for l in arr]
        inv_cdf = [l for arr in inv_cdf for l in arr]

    return integral, inv_cdf