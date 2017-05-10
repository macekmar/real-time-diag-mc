#include "./solver_core.hpp"
#include "./moves.hpp"
#include <gsl/gsl_integration.h>
#include <triqs/det_manip.hpp>

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;
using triqs::arrays::range;


// ------------ The main class of the solver ------------------------
solver_core::solver_core(solve_parameters_t const& params)
   : qmc(params.random_name, params.random_seed, 1.0, params.verbosity, false), // first_sign is not used
     params(params) {

 compilation_time_stamp(48);

#ifdef REGISTER_CONFIG
 if (mpi::communicator().rank() == 0)
  std::cout << "/!\ Configuration Registration ON" << std::endl << std::endl;
#endif

 if (params.interaction_start < 0) TRIQS_RUNTIME_ERROR << "interaction_time must be positive";
 if (params.max_perturbation_order - params.min_perturbation_order + 1 < 2)
  TRIQS_RUNTIME_ERROR << "The range of perturbation orders must cover at least 2 orders";

 // boundaries
 t_max = *std::max_element(params.measure_times.begin(), params.measure_times.end());
 t_max = std::max(t_max, std::get<1>(params.right_input_points[0]));
 t_min = *std::min_element(params.measure_times.begin(), params.measure_times.end());
 t_min = std::min(t_min, std::get<1>(params.right_input_points[0]));

 // kernel binning
 kernels_all = array<dcomplex, 3>(params.max_perturbation_order + 1, params.nb_bins, 2);
 kernels = array<dcomplex, 3>(params.max_perturbation_order + 1, params.nb_bins, 2);
 kernels_binning =
     KernelBinning(-params.interaction_start, t_max, params.nb_bins, params.max_perturbation_order,
                   true); // TODO: it should be defined in measure

 // rank of the Green's function to calculate. For now only 1 is supported.
 if (params.right_input_points.size() % 2 == 0)
  TRIQS_RUNTIME_ERROR << "There must be an odd number of right input points";
 if (params.right_input_points.size() > 1)
  TRIQS_RUNTIME_ERROR << "For now only rank 1 Green's functions are supported.";
 rank = params.right_input_points.size() / 2;

 taup = make_keldysh_contour_pt(params.right_input_points[0]);

 // make tau list from left input points lists
 tau_array = array<keldysh_contour_pt, 2>(params.measure_times.size(), params.measure_keldysh_indices.size());
 if (tau_array.is_empty()) TRIQS_RUNTIME_ERROR << "No left input point !";

 keldysh_contour_pt tau;
 for (int i = 0; i < params.measure_times.size(); ++i) {
  for (int j = 0; j < params.measure_keldysh_indices.size(); ++j) {
   tau = {params.measure_state, params.measure_times[i], params.measure_keldysh_indices[j]};
   tau_array(i, j) = tau;
  }
 }

 // results arrays
 pn = array<long, 1>(params.max_perturbation_order + 1);     // pn for this process
 pn_all = array<long, 1>(params.max_perturbation_order + 1); // gather pn for all processes
 pn() = 0;
 pn_all() = 0;

 sn = array<dcomplex, 3>(params.max_perturbation_order + 1, params.measure_times.size(),
                         params.measure_keldysh_indices.size()); // multidim array of sn for this process
 sn_all = array<dcomplex, 3>(params.max_perturbation_order + 1, params.measure_times.size(),
                             params.measure_keldysh_indices.size()); // multidim array of sn for all processes
 sn() = 0;
 sn_all() = 0;
};

// --------------------------------
// TODO: put the green functions in the parameters dictionnary so as to construct in one step only
void solver_core::set_g0(gf_view<retime, matrix_valued> g0_lesser,
                         gf_view<retime, matrix_valued> g0_greater) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status > not_ready) TRIQS_RUNTIME_ERROR << "Green functions already set up. Cannot change.";

 // non interacting Green function
 green_function_alpha =
     g0_keldysh_alpha_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max};
 green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}};
 config = Configuration(green_function_alpha, tau_array(0, 0), taup, params.max_perturbation_order,
                        params.singular_thresholds);

 // order zero values
 auto gf_map = map([this](keldysh_contour_pt tau) { return green_function(tau, taup); });
 g0_array = make_matrix(gf_map(tau_array));

 // Register moves and measurements
 if (params.w_ins_rem > 0) {
  qmc.add_move(moves::insert{&config, &params, t_max, qmc.get_rng()}, "insertion", params.w_ins_rem);
  qmc.add_move(moves::remove{&config, &params, t_max, qmc.get_rng()}, "removal", params.w_ins_rem);
 }
 if (params.w_dbl > 0) {
  qmc.add_move(moves::insert2{&config, &params, t_max, qmc.get_rng()}, "insertion2", params.w_dbl);
  qmc.add_move(moves::remove2{&config, &params, t_max, qmc.get_rng()}, "removal2", params.w_dbl);
 }
 if (params.w_shift > 0) {
  qmc.add_move(moves::shift{&config, &params, t_max, qmc.get_rng()}, "shift", params.w_shift);
 }
 if (params.method == 4) {
  qmc.add_move(moves::weight_swap{&config, &params, t_max, qmc.get_rng()}, "weight swap",
               params.w_weight_swap);
  qmc.add_move(moves::weight_shift{&config, &params, t_max, qmc.get_rng()}, "weight shift",
               params.w_weight_shift);
 }

 if (params.method == 0) {
  if (size(tau_array) > 1)
   TRIQS_RUNTIME_ERROR << "Trying to use a singlepoint measure with multiple input point";
  qmc.add_measure(WeightSignMeasure(&config, &pn, &pn_all, &sn, &sn_all), "Weight sign measure");
 } else if (params.method == 4) {
  qmc.add_measure(TwoDetCofactMeasure(&config, &kernels_binning, &pn, &pn_all, &sn, &sn_all, &tau_array,
                                      &g0_array, green_function, params.interaction_start + t_max),
                  "Cofact measure");
 } else if (params.method == 5) {
  qmc.add_measure(TwoDetKernelMeasure(&config, &kernels_binning, &pn, &pn_all, &sn, &sn_all, &kernels,
                                      &kernels_all, &tau_array, &g0_array, green_function,
                                      params.interaction_start + t_max),
                  "Kernel measure");
 } else {
  TRIQS_RUNTIME_ERROR << "Cannot recognise the method ID";
 }

 status = ready;
};

// --------------------------------
std::function<bool()> solver_core::make_callback(int time_in_seconds) {
 auto clock_callback = triqs::utility::clock_callback(time_in_seconds);
 return [clock_callback]() {
  MPI_Barrier(MPI_COMM_WORLD);
  return clock_callback();
 };
};

// --------------------------------
int solver_core::run(const int nb_cycles, const bool do_measure, const int max_time = -1) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set !";
 // order zero case
 if (params.max_perturbation_order == 0)
  TRIQS_RUNTIME_ERROR << "Order zero cannot run, use order_zero method instead";

 mpi::communicator world;
 std::cout << "reached start run barrier" << std::endl;
 MPI_Barrier(MPI_COMM_WORLD);

 int run_status;

 // warmup if run has not started yet
 if (status == ready) {
  status = running;
  solve_duration = 0;
  //if (world.rank() == 0) std::cout << "Warming up... " << std::flush;
  //run_status =
      //qmc.run(params.n_warmup_cycles, params.length_cycle, triqs::utility::clock_callback(-1), false);
  if (run_status == 2) return finish(run_status); // Received a signal: abort
  //if (world.rank() == 0) std::cout << "done" << std::endl;
 }

 // accumulate
 std::cout << "Accumulate..." << std::endl;
 run_status = qmc.run(nb_cycles, params.length_cycle, triqs::utility::clock_callback(max_time), do_measure);

 // Collect results
 std::cout << "Collecting results... " << std::flush;
 qmc.collect_results(world);
 std::cout << "done" << std::endl << std::endl;

 solve_duration = solve_duration + qmc.get_duration();
 solve_duration_all = mpi::mpi_all_reduce(solve_duration);

 array<double, 1> weight_sum = config.get_weight_sum();
 weight_sum = mpi::mpi_all_reduce(weight_sum);
 array<long, 1> nb_values = config.get_nb_values();
 nb_values = mpi::mpi_all_reduce(nb_values);
 array<double, 1> weight_avg = weight_sum / nb_values;

 array<long, 1> nb_cofact = mpi::mpi_all_reduce(config.nb_cofact);
 array<long, 1> nb_inverse = mpi::mpi_all_reduce(config.nb_inverse);

 // print acceptance rates
 if (world.rank() == 0) {
  std::cout << "Duration: " << solve_duration << " seconds" << std::endl;
  std::cout << "Nb of measures: " << get_nb_measures_all() << std::endl;
  std::cout << "Acceptance rates of node 0:" << std::endl;
  double total_weight = 0;
  double total_rate = 0;
  double move_weight;
  for (auto const& x : qmc.get_acceptance_rates()) { // x.first = key, x.second = value

   if (x.first == "insertion" or x.first == "removal")
    move_weight = params.w_ins_rem;
   else if (x.first == "insertion2" or x.first == "removal2")
    move_weight = params.w_dbl;
   else if (x.first == "shift")
    move_weight = params.w_shift;
   else if (x.first == "weight swap")
    move_weight = params.w_weight_swap;
   else if (x.first == "weight shift")
    move_weight = params.w_weight_shift;
   else
    move_weight = 0;
   if (move_weight > 0) {
    total_weight += move_weight;
    total_rate += move_weight * x.second;
   }

   std::cout << "> " << x.first << " (" << move_weight << "): " << x.second << std::endl;
  }
  std::cout << "> All moves: " << total_rate / total_weight << std::endl;
  std::cout << "Attempted config weight average:" << std::endl << weight_avg << std::endl;
  // std::cout << "Weight offsets:" << std::endl << config.weight_offsets << std::endl;
  std::cout << "cofact vs inverse : " << nb_cofact << " / " << nb_inverse << std::endl;
  std::cout << "regen ratios : 0=" << config.matrices[0].get_regen_ratio()
            << ", 1=" << config.matrices[1].get_regen_ratio() << std::endl;
  std::pair<double, double> regen_stats;
  for (int a : {0, 1}) {
   regen_stats = config.matrices[a].get_error_stats();
   std::cout << "regen errors " << a << " : avg=" << regen_stats.first << ", std=" << std::sqrt(regen_stats.second)
             << std::endl;
  }
  std::cout << std::endl;
 }
 std::cout << "End run " << std::flush;

 if (run_status != 1) return finish(run_status);

 return run_status;
}

// --------------------------------
void solver_core::compute_sn_from_kernels() {
 std::cout << "Computing sn from kernels..." << std::endl;
 keldysh_contour_pt tau;
 for (int order = 0; order < first_dim(sn); ++order) { // for each order
  for (int i = 0; i < second_dim(sn); ++i) {       // for each tau (time)
   for (int a = 0; a < third_dim(sn); ++a) {       // for each tau (keldysh index)
    tau = tau_array(i, a);
    auto gf_map = map([&](keldysh_contour_pt alpha) { return green_function(tau, alpha); });
    auto gf_tau_alpha = gf_map(kernels_binning.coord_array());
    sn(order, i, a) = sum(gf_tau_alpha * kernels(order, ellipsis()));
    sn_all(order, i, a) = sum(gf_tau_alpha * kernels_all(order, ellipsis()));
   }
  }
 }
}

// --------------------------------
int solver_core::finish(const int run_status) {
 if (run_status == 1) TRIQS_RUNTIME_ERROR << "finish cannot be called when the run is not finished";

 if (run_status == 2) status = aborted; // Received a signal: abort

 if (run_status == 0 and mpi::communicator().rank() == 0)
  std::cout << "Run finished" << std::endl << std::endl;
 if (run_status == 2 and mpi::communicator().rank() == 0)
  std::cout << "Run aborted" << std::endl << std::endl;

 return run_status;
}

// --------------------------------
struct integrand_params {
 g0_keldysh_t green_function;
 keldysh_contour_pt tau;
 keldysh_contour_pt taup;

 integrand_params(g0_keldysh_t green_function, int state, int k_index, keldysh_contour_pt taup)
    : green_function(green_function), tau{state, 0., k_index}, taup(taup){};
};

double abs_g0_keldysh_t_inputs(double t, void* _params) {
 auto params = static_cast<integrand_params*>(_params);
 keldysh_contour_pt tau = params->tau;
 tau.t = t;
 return std::abs(params->green_function(tau, params->taup));
};

std::tuple<double, array<dcomplex, 2>> solver_core::order_zero() {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set!";
 if (mpi::communicator().rank() == 0) std::cout << "Order zero calculation... " << std::flush;
 double c0 = 0;
 double c0_error = 0;

 if (params.method == 4) {
  // Uses GSL integration
  // (https://www.gnu.org/software/gsl/manual/html_node/Numerical-integration-examples.html)
  gsl_function F;
  F.function = &abs_g0_keldysh_t_inputs;
  auto w = gsl_integration_cquad_workspace_alloc(10000);
  size_t nb_evals;
  double value;
  double value_error;

  for (int a : {0, 1}) {
   integrand_params int_params(green_function, 0, a, taup);
   F.params = &int_params;
   gsl_integration_cquad(&F,                        // function to integrate
                         -params.interaction_start, // lower boundary
                         t_max,                     // upper boundary
                         1e-6,                      // absolute error
                         1e-6,                      // relative error
                         w,                         // workspace
                         &value,                    // result
                         &value_error,              // output absolute error
                         &nb_evals);                // number of function evaluation
   c0 += value;
   c0_error += value_error;
  }

  gsl_integration_cquad_workspace_free(w);
 } else if (params.method == 5) {
  c0 = 1.;
 } else { // singlepoint methods only
  c0 = abs(g0_array(0, 0));
 }
 array<dcomplex, 2> s0 = g0_array / c0;
 if (mpi::communicator().rank() == 0) {
  std::cout << "done" << std::endl;
  std::cout << "c0 = " << c0 << " error = " << c0_error << std::endl << std::endl;
 }
 return std::tuple<double, array<dcomplex, 2>>{c0, s0};
}
