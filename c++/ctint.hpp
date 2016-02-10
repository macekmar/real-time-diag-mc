#include <triqs/gfs.hpp>
#include "./qmc_data.hpp"
#include "g0_semi_circ.hpp"

template <typename T> using view_t = typename T::view_type;

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;

class ctint_solver {

 g0_t g0_lesser, g0_greater;

 public:
//FIXME ctint_solver(g0_t::view_type g0_lesser, g0_t::view_type g0_greater) : g0_lesser(g0_lesser), g0_greater(g0_greater){};
 ctint_solver(gf_view<retime, scalar_valued> g0_lesser, gf_view<retime, scalar_valued> g0_greater) : g0_lesser(g0_lesser), g0_greater(g0_greater){};

 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     std::pair<array<double, 1>, array<double, 1>>
     solve(solve_parameters_t const& params);
};
