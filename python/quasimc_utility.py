import numpy as np

def distribute_u(u, order, t_min, N_vec, N_proc):
    # Generated points are split into the sets of demanded number of points.
    # If we want results for 100 and 1000 points, points[0] hold first 100
    # points and points[1] the remaining 900.
    # points[k] are then split into processes. They are padded with zero,
    # so that their length is divisible by N_proc
    # (i + j*world_size)-th point, where j = 1,...,len(points[k])/N_proc,  
    # goes to the i-th process
    points = []
    N_true = []
    for i in range(len(N_vec)-1):
        p = np.copy(u[N_vec[i]:N_vec[i+1]])
        # Remove points outside of the domain
        inds = np.where(np.all(p > t_min, axis=1))[0]
        p = p[inds]
        N_true.append(p.shape[0])

        sub_len = np.int(np.ceil(np.true_divide(len(p), N_proc)))
        p.resize(sub_len*N_proc, order)
        p = p.reshape((sub_len, N_proc, order))
        p = np.array_split(p, N_proc, axis=1)
        for j, _ in enumerate(p):
            p[j] = p[j][:,0,...] # 3D array, 2nd dim is 1, problem if 1st is 1

        points.append(p)

    return N_true, points


def distribute_u_complex(u, order, t_min, N_vec, N_proc):
    points = []
    N_true = []
    # Find all points in the domain ...
    all_inds = np.where(np.all(u > t_min, axis=1))[0]
    # First point may not be in the domain
    if all_inds[0] != 0:
        all_inds = np.insert(all_inds, 0, 0)
    # N_vec is [0, N1, N2, ...]
    N_first = 0
    for i in range(len(N_vec)-1):
        # ... and take N_i of them
        p = np.copy(u[ all_inds[N_vec[i]:N_vec[i+1]] ])
        # Explanation of the three lines below:
        # all_inds contain indices of the point inside the domain, thus when we
        # take all_inds[:N] indices we have N points inside the domain
        # the last index is all_inds[N-1]
        # This is the minimum number of points we have to generate to get N
        # points inside the domain. Because indices for arrays start with 0,
        # we add 1.
        # We are interested in the number of points between two N.
        N_last = all_inds[N_vec[i+1]-1] + 1
        N_true.append(N_last - N_first) # N_true is the number of generated points
        N_first = N_last
        # Remove points outside of the domain
        inds = np.where(np.all(p > t_min, axis=1))[0]
        p = p[inds]
        
        sub_len = np.int(np.ceil(np.true_divide(len(p), N_proc)))
        p.resize(sub_len*N_proc, order)
        p = p.reshape((sub_len, N_proc, order))
        p = np.array_split(p, N_proc, axis=1)
        for j, _ in enumerate(p):
            p[j] = p[j][:,0,...] # 3D array, 2nd dim is 1, problem if 1st is 1

        points.append(p)

    return N_true, points