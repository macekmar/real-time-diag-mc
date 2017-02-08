import numpy as np

def analytic_order1(g0, epsilon, times, t2=0.0, delta=1e-6):
    g0_lesser_plus,  g0_greater_plus  = g0(epsilon + 0.5 * delta)
    g0_lesser_minus, g0_greater_minus = g0(epsilon - 0.5 * delta)
    density_value = -1j * g0(epsilon)[0](t2) # g0_lesser

    deriv_lesser  = np.array([(g0_lesser_plus(t-t2) - g0_lesser_minus(t-t2)) / delta for t in times],
                             dtype=complex)
    deriv_greater = np.array([(g0_greater_plus(t-t2) - g0_greater_minus(t-t2)) / delta for t in times], 
                             dtype=complex)

    return density_value * deriv_lesser, density_value * deriv_greater

if __name__ == '__main__':
    from ctint_keldysh import make_g0_semi_circular
    from pytriqs.archive import *

    #-----------------
    times = np.linspace(-40.0, 0.0, 101)
    o1_lesser, o1_greater = analytic_order1(lambda epsilon : make_g0_semi_circular(beta=200,
                                                                                  Gamma=0.5,
                                                                                  tmax_gf0=100.0,
                                                                                  Nt_gf0=25000,
                                                                                  epsilon_d=epsilon,
                                                                                  muL=0.0,
                                                                                  muR=0.0),
                                            0.5,
                                            times)

    with HDFArchive("order1_params1.ref.h5", 'w') as ar:
        ar["times"] = times
        ar["t2"] = 0.0
        ar["o1_less"] = o1_lesser[:, 0, 0]
        ar["o1_grea"] = o1_greater[:, 0, 0]

    #-----------------
    times = np.linspace(-40.0, 0.0, 101)
    o1_lesser, o1_greater = analytic_order1(lambda epsilon : make_g0_semi_circular(beta=200,
                                                                                  Gamma=1.,
                                                                                  tmax_gf0=100.0,
                                                                                  Nt_gf0=25000,
                                                                                  epsilon_d=epsilon,
                                                                                  muL=0.5,
                                                                                  muR=0.0),
                                            0.25,
                                            times)

    with HDFArchive("order1_params2.ref.h5", 'w') as ar:
        ar["times"] = times
        ar["t2"] = 0.0
        ar["o1_less"] = o1_lesser[:, 0, 0]
        ar["o1_grea"] = o1_greater[:, 0, 0]
