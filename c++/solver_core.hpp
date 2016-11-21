#include <triqs/gfs.hpp>
#include "./qmc_data.hpp"

template <typename T> using view_t = typename T::view_type;

// ------------ The main class of the solver -----------------------

using namespace triqs::gfs;

class solver_core {

 g0_t g0_lesser, g0_greater;
 double _solve_duration = 0;

 public:
 // FIXME change type of arguments after olivier fixes wrapper
 solver_core(gf_view<retime, matrix_valued> g0_lesser, gf_view<retime, matrix_valued> g0_greater)
    : g0_lesser(g0_lesser), g0_greater(g0_greater){};

 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     std::pair<std::pair<array<double, 1>, array<double, 1>>, std::pair<array<double, 1>, array<double, 1>>>
     solve(solve_parameters_t const& params);
 
 double get_solve_duration() const {return _solve_duration;}
};
