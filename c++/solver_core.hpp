#pragma once
#include "./measure.hpp"
#include "./qmc_data.hpp"
#include <assert.h>
#include <triqs/mc_tools.hpp>

#include <triqs/gfs.hpp>

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
 std::vector<keldysh_contour_pt> creation_pts;
 std::vector<keldysh_contour_pt> annihila_pts;
 array<keldysh_contour_pt, 2> tau_array;
 array<dcomplex, 2> g0_array;
 double t_max;
 double qmc_duration = 0;
 double cum_qmc_duration = 0;
 Status status = not_ready;
 KernelBinning kernels_binning;
 array<dcomplex, 3> kernels;
 array<dcomplex, 3> kernel_diracs;
 array<long, 3> nb_kernels;
 array<long, 1> pn;
 array<dcomplex, 3> sn;

 int finish(const int run_status);

 public:
 // FIXME change type of arguments after olivier fixes wrapper
 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     solver_core(solve_parameters_t const& params);

 void set_g0(gf_view<retime, matrix_valued> g0_lesser, gf_view<retime, matrix_valued> g0_greater);

 std::tuple<double, array<dcomplex, 2>> order_zero();

 int run(const int nb_cycles, const bool do_measure, const int max_time);
 int run(const int nb_cycles, const bool do_measure) { return run(nb_cycles, do_measure, -1); };

 void compute_sn_from_kernels();
 void collect_results(int nb_partitions);

 // getters
 double get_qmc_duration() const { return cum_qmc_duration; }
 long get_nb_measures() const { return sum(pn); }
 std::vector<std::vector<double>> get_config_list() const { return config.config_list; }
 std::vector<int> get_config_weight() const { return config.config_weight; }
 array<long, 1> get_pn() const { return pn; }
 array<dcomplex, 3> get_sn() const { return sn; }
 array<dcomplex, 3> get_kernels() const { return kernels / kernels_binning.get_bin_length(); }
 array<dcomplex, 3> get_kernel_diracs() const { return kernel_diracs; }
 array<long, 3> get_nb_kernels() const { return nb_kernels; }
 array<double, 1> get_bin_times() const { return kernels_binning.get_bin_times(); }
 array<double, 1> get_dirac_times() const { return kernels_binning.get_dirac_times(); }
 double get_U() const { return params.U; }
 int get_max_order() const { return params.max_perturbation_order; }
};
