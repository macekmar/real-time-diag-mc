# distutils: language = c++
# distutils: sources = ./c++/sobol_c.cpp

from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "./c++/sobol_c.hpp":
    pair[int, vector[double]] i4_sobol (int dim_num, int seed)

def sobol_single_value(dim_num, seed):
    return i4_sobol(dim_num, seed)

def sobol_generator(dim_num, start):
    seed = 0
    for _ in range(start):
        seed, _ = i4_sobol_py(dim_num, seed)

    while True:
        seed, values = i4_sobol_py(dim_num, seed)
        yield values


