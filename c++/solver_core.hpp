#pragma once
#include "./measure.hpp"
#include "./qmc_data.hpp"
#include <assert.h>
#include <triqs/mc_tools.hpp>

#include <triqs/gfs.hpp>

#define STRING2(x) #x
#define STRING(x) STRING2(x)

void compilation_time_stamp(int node_size) {
 int rank;
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef COMPILATION_TIMESTAMP
 if (rank % node_size == 0)
  std::cout << "Rank " << rank << " : " <<  STRING(COMPILATION_TIMESTAMP) << std::endl;
#endif
};

// using namespace triqs::gfs;
using namespace triqs::arrays;

auto size_fold = fold([](size_t r, keldysh_contour_pt x) { return r + 1; });
template <typename ArrayType> size_t size(ArrayType some_array) { return size_fold(some_array, 0); }

// ------------ The main class of the solver -----------------------

enum Status { aborted, not_ready, ready, running };

class solver_core {

 g0_keldysh_alpha_t green_function_alpha;
 g0_keldysh_t green_function;
 Configuration config;
 solve_parameters_t params;
 triqs::mc_tools::mc_generic<dcomplex> qmc;
 keldysh_contour_pt taup;
 array<keldysh_contour_pt, 2> tau_array;
 array<dcomplex, 2> g0_array;
 double t_max, t_min;
 int rank;
 double solve_duration = 0;
 double solve_duration_all = 0;
 Status status = not_ready;
 KernelBinning kernels_binning;
 array<dcomplex, 3> kernels;
 array<dcomplex, 3> kernels_all;
 array<long, 1> pn;
 array<long, 1> pn_all;
 array<dcomplex, 3> sn;
 array<dcomplex, 3> sn_all;

 int finish(const int run_status);

 std::function<bool()> make_callback(int time_in_seconds);

 public:
 // FIXME change type of arguments after olivier fixes wrapper
 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     solver_core(solve_parameters_t const& params);

 void set_g0(gf_view<retime, matrix_valued> g0_lesser, gf_view<retime, matrix_valued> g0_greater);

 std::tuple<double, array<dcomplex, 2>> order_zero();

 int run(const int nb_cycles, const bool do_measure, const int max_time);
 int run(const int nb_cycles, const bool do_measure) { return run(nb_cycles, do_measure, -1); };

 // getters
 double get_solve_duration() const { return solve_duration; }
 double get_solve_duration_all() const { return solve_duration_all; }
 long get_nb_measures() const { return sum(pn); }
 long get_nb_measures_all() const { return sum(pn_all); }
 std::vector<std::vector<double>> get_config_list() const { return config.config_list; }
 std::vector<int> get_config_weight() const { return config.config_weight; }
 array<long, 1> get_pn() const { return pn; }
 array<long, 1> get_pn_all() const { return pn_all; }
 array<dcomplex, 3> get_sn() const { return sn; }
 array<dcomplex, 3> get_sn_all() const { return sn_all; }
 array<dcomplex, 3> get_kernels() const { return kernels; }
 array<dcomplex, 3> get_kernels_all() const { return kernels_all; }
 array<long, 3> get_nb_kernels() const { return kernels_binning.get_nb_values(); }
};
