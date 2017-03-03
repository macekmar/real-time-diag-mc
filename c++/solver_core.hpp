#include "./measure.hpp"
#include "./qmc_data.hpp"
#include <triqs/mc_tools.hpp>
#include <memory>

#include <triqs/gfs.hpp>

// template <typename T> using view_t = typename T::view_type;
using namespace triqs::gfs;

// ------------ The main class of the solver -----------------------

enum Status { aborted, not_ready, ready, running };

class solver_core {

 g0_keldysh_t green_function;
 std::shared_ptr<Integrand> integrand = nullptr;
 solve_parameters_t params;
 triqs::mc_tools::mc_generic<dcomplex> qmc;
 keldysh_contour_pt taup;
 std::vector<keldysh_contour_pt> tau_list;
 std::vector<std::size_t> shape_tau_array;
 array<dcomplex, 1> g0_values = array<dcomplex, 1>();
 array<dcomplex, 1> prefactor;
 double t_max;
 int rank;
 int op_to_measure_spin; // spin of the operator to measure. Not needed when up/down symmetry. Is used to know which determinant
                         // is the big one.
 double solve_duration = 0;
 int nb_measures;
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_weight;
 Status status = not_ready;
 array<int, 1> pn;
 array<dcomplex, 2> sn;
 array<dcomplex, 3> sn_array;
 array<double, 1> pn_errors;
 array<double, 1> sn_errors;

 int finish(const int run_status);

 Measure* create_measure(const int method, const Weight* weight);

 array<dcomplex, 3> reshape_sn(array<dcomplex, 2>* sn_list);
 array<dcomplex, 2> reshape_sn(array<dcomplex, 1>* sn_list);

 public:
 // FIXME change type of arguments after olivier fixes wrapper
 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     solver_core(solve_parameters_t const& params);

 void set_g0(gf_view<retime, matrix_valued> g0_lesser, gf_view<retime, matrix_valued> g0_greater);

 std::tuple<double, array<dcomplex, 2>> order_zero();

 int run(const int max_time, const int max_measures);
 int run(const int max_time) {return run(max_time, -1);};
 int run() {return run(-1, -1);};

 // getters
 double get_solve_duration() const { return solve_duration; }
 int get_nb_measures() const { return nb_measures; }
 std::vector<std::vector<double>> get_config_list() const { return config_list; }
 std::vector<int> get_config_weight() const { return config_weight; }
 array<int, 1> get_pn() const { return pn; }
 array<dcomplex, 3> get_sn() const { return sn_array; }
 array<double, 1> get_pn_errors() const { return pn_errors; }
 array<double, 1> get_sn_errors() const { return sn_errors; }
};
