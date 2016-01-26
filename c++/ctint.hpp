#include <triqs/gfs.hpp>
#include "./qmc_data.hpp"

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

class ctint_solver {

 gf0_t g0_lesser, g0_greater;

 public:
 ctint_solver(double beta,
              double mu,
              int n_freq,
              double t_min,
              double t_max,
              int L,
              int Lk);

 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     std::pair<array<double, 1>, array<double, 1>>
     solve(solve_parameters_t const& params);
};

