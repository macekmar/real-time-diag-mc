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
            #p[j] = p[j].squeeze() # 3D array, 2nd dim is 1, problem if 1st is 1
            p[j] = p[j][:,0,...]

        points.append(p)

    return N_true, points

