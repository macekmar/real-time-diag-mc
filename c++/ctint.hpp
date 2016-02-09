#include <triqs/gfs.hpp>
#include "./qmc_data.hpp"

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;

class ctint_solver {

 g0_t g0_lesser, g0_greater;

 public:
 ctint_solver(g0_t g0_lesser, g0_t g0_greater) : g0_lesser(g0_lesser), g0_greater(g0_greater){};

 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     std::pair<array<double, 1>, array<double, 1>>
     solve(solve_parameters_t const& params);
};
