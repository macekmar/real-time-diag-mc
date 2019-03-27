import sobol_seq
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import PchipInterpolator
from mpi4py import MPI

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
    vals = 1
    for i in range(order-1):
        arg = (t[:,i:i+1] - t[:,i+1:i+2])
        vals *= funs[-(order-i)](arg)
        #vals *= self.get(arg)**(1+2*((i+1)%2)) # For symmetric case (params Corentin 2)
    vals *= funs[-(order-i)](t[:,-1:])
    return vals

def generate_u(inv_cdf, interaction_start, N_samples, dim, N_skip=0):
    """Generates u distributed by spring model.
    It generates N_samples v-variables. Some u-variables (true variables) 
    can lie outside of the domain [-interaction_start, 0]^order."""
    v = sobol_seq.i4_sobol_generate(dim, N_samples, skip=N_skip+1)
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
    It returns N_samples in the domain"""
     # This came from a fit
    N_batch = int(1.02/np.exp(-(dim-1.1325)**2/7.6**2)*N_samples)
    itr = 0
    while N < N_samples:
        # Inverse transform sampling
        v = sobol_seq.i4_sobol_generate(dim, N_batch, skip=N_skip + N+1)
        u = np.zeros_like(v)
        for i in range(-dim, 0):
            u[:,i] = inv_cdf[i](v[:,i])
        # Back to u from v
        for i in range(2, dim+1):
            u[:,-i] = u[:,-i] + u[:,-i+1]
        # Reject u out of t_min
        inds = np.where(np.all(u > interaction_start, axis=1))[0]
        if N == 0:
            u_samples = u[inds,:]
        else:
            u_samples = np.vstack((u_samples, u[inds,:]))

        N += len(inds)

        itr += 1
        #print("Loop %d, added: %d, done %d" % (itr, len(inds), N_batch))
        N_batch = int(1.02/np.exp(-(dim-1.1325)**2/7.65**2)*(N_samples-N))
    # At most we reject 10 % of samples!
    return u_samples[:N_samples,:]


def _calculate_inv_cdf(fun, t_min, t_max=0, Nt=201):
    """Calculates inverse CDF for a nonnormalized function."""
    u_lin = np.linspace(t_min, t_max, Nt)
    fun_val = np.abs(fun(u_lin[:, np.newaxis])) # Newaxis is necessary for the get function
    cdf = cumtrapz(fun_val, u_lin, initial=0)
    cdf = cdf/cdf[-1]
    return PchipInterpolator(cdf, u_lin)

def calculate_inv_cdfs(model, t_min, t_max=0, Nt=201):
    
    world = MPI.COMM_WORLD
    # Split
    if world.rank == 0:
        print('Calculating inverse CDF')
        funs = model
        funs = np.array_split(funs, world.size)
    else:
        funs = None

    funs = world.scatter(funs, root=0)
    inv_cdf = [None for i in range(len(funs))]
    # Calculate
    for i, f in enumerate(funs):
        inv_cdf[i] = _calculate_inv_cdf(f, t_min=t_min, t_max=t_max, Nt=Nt)
    # Gather
    inv_cdf = world.gather(inv_cdf, root=0)
    if world.rank == 0:
        inv_cdf = [l for arr in inv_cdf for l in arr]

    return inv_cdf