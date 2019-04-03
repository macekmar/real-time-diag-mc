#include <utility>
#include <triqs/arrays.hpp>
using namespace triqs::arrays;

int i4_bit_hi1 ( int n );
int i4_bit_lo0 ( int n );
std::pair<int, array<double, 1>> i4_sobol (int dim_num, int seed);
array<double, 2> i4_sobol_generate ( int m, int n, int skip );
