#pragma once
#include "./random_vertex_gen.hpp"
#include "./measure.hpp"
#include "./qmc_data.hpp"
#include "./model.hpp"
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
 Model model;

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
 array<long, 1> get_pn() const;
 array<dcomplex, 1> get_sn() const;
 array<dcomplex, 4> get_kernels() const;
 array<dcomplex, 4> get_kernel_diracs() const;
 array<long, 4> get_nb_kernels() const;
 array<double, 1> get_bin_times() const { return kernels_binning.get_bin_times(); }
 array<double, 1> get_dirac_times() const { return kernels_binning.get_dirac_times(); }
 std::vector<double> get_U() const { return params.U; }
 int get_max_order() const { return params.max_perturbation_order; }

 // Importance sampling part
 std::vector<dcomplex> evaluate_importance_sampling(std::vector<timec_t> times_l, bool do_measure = false);
 dcomplex evaluate_model(std::vector<timec_t> times_l_vec);
 timec_t inverse_cdf(int i, timec_t l) {return model.inverse_cdf(i, l);};
 void collect_sampling_weights(int dummy);
 void set_model(std::vector<std::vector<double>> intervals, std::vector<std::vector<std::vector<double>>> coeff);
 
 std::vector<double> l_to_u(std::vector<timec_t> times_l) {return model.l_to_u(times_l);};
 std::vector<double> u_to_l(std::vector<timec_t> times_u) {return model.u_to_l(times_u);};
 std::vector<double> l_to_v(std::vector<timec_t> times_l) {return model.l_to_v(times_l);};
 std::vector<double> v_to_l(std::vector<timec_t> times_v) {return model.v_to_l(times_v);};
 std::vector<double> v_to_u(std::vector<timec_t> times_v) {return model.v_to_u(times_v);};
 std::vector<double> u_to_v(std::vector<timec_t> times_u) {return model.u_to_v(times_u);};
 
};
