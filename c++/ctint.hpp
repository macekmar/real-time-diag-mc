#include <triqs/gfs.hpp>
#include "./parameters.hpp"

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

class ctint_solver {

 gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail> g0_lesser, g0_greater;
 array<double, 2> cn_sn = array<double, 2>(2, 20); ///< measurement of c_n and s_n at different n
 mpi::communicator world;

 public:
 ctint_solver(double beta,
              double mu,
              int n_freq,
              double t_min,
              double t_max,
              int L,
              int Lk);

 array_view <double,2> CnSn() { return cn_sn;}
 
 // ???
 double c0()   { return imag(g0_lesser(mindex(0,0,0),0.0)); } ///< non interacting charge
 
 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
 void solve(solve_parameters_t const & params);
};

