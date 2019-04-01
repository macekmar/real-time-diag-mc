#include <triqs/arrays.hpp>
#include <tuple>

using namespace triqs::arrays;

int i4_bit_hi1 ( int n );
int i4_bit_lo0 ( int n );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
std::tuple<int, array<double, 1>> i4_sobol (int dim_num, int seed);
array<double, 2> i4_sobol_generate ( int m, int n, int skip );

double r4_abs ( double x );
int r4_nint ( double x );
double r4_uniform_01 ( int seed );

int tau_sobol ( int dim_num );
void timestamp ( );

