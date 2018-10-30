#pragma once
#include "./random_vertex_gen.hpp"
#include "./measure.hpp"
#include "./qmc_data.hpp"
#include <triqs/mc_tools.hpp>

#include <triqs/gfs.hpp>


// ------------ The main class of the solver -----------------------

enum Status { aborted, not_ready, ready, running };

class solver_core {

 Configuration config;
 solve_parameters_t params;
 triqs::mc_tools::mc_generic<dcomplex> qmc;
 std::unique_ptr<RandomVertexGenerator> rvg;
 double qmc_duration = 0;
 double cum_qmc_duration = 0;
 Status status = not_ready;
 KernelBinning kernels_binning;
 array<dcomplex, 4> kernels;
 array<dcomplex, 4> kernel_diracs;
 array<long, 4> nb_kernels;
 array<long, 1> pn;
 array<dcomplex, 1> sn;

 int finish(const int run_status);

 public:
 // FIXME change type of arguments after olivier fixes wrapper
 TRIQS_WRAP_ARG_AS_DICT // Wrap the solver parameters as a ** call in python with the clang & c++2py tool
     solver_core(solve_parameters_t const& params);

 void set_g0(triqs::gfs::gf_view<triqs::gfs::retime, triqs::gfs::matrix_valued> g0_lesser,
             triqs::gfs::gf_view<triqs::gfs::retime, triqs::gfs::matrix_valued> g0_greater);

 int run(const int nb_cycles, const bool do_measure, const int max_time);
 int run(const int nb_cycles, const bool do_measure) { return run(nb_cycles, do_measure, -1); };

 bool collect_results(int size_partition);

 dcomplex evaluate_qmc_weight(std::vector<std::tuple<orbital_t, orbital_t, timec_t>> vertices);

 // getters
 double get_qmc_duration() const { return cum_qmc_duration; }
 long get_nb_measures() const { return sum(pn); }
 std::vector<std::vector<double>> get_config_list() const { return config.config_list; }
 std::vector<int> get_config_mult() const { return config.config_mult; }
 std::vector<dcomplex> get_config_weight() const { return config.config_weight; }
 array<long, 1> get_pn() const { return pn; }
 array<dcomplex, 1> get_sn() const { return sn; }
 array<dcomplex, 4> get_kernels() const { return kernels / kernels_binning.get_bin_length(); }
 array<dcomplex, 4> get_kernel_diracs() const { return kernel_diracs; }
 array<long, 4> get_nb_kernels() const { return nb_kernels; }
 array<double, 1> get_bin_times() const { return kernels_binning.get_bin_times(); }
 array<double, 1> get_dirac_times() const { return kernels_binning.get_dirac_times(); }
 std::vector<double> get_U() const { return params.U; }
 int get_max_order() const { return params.max_perturbation_order; }
};
