#include "./solver_core.hpp"
#include "./accumulator.hpp"
#include "./moves.hpp"
#include "./weight.hpp"
#include <chrono>
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
     params(params),
     nb_measures(0) {

#ifdef REGISTER_CONFIG
 if (triqs::mpi::communicator().rank() == 0) std::cout << "/!\ Configuration Registration ON" << std::endl << std::endl;
#endif

 if (params.interaction_start < 0) TRIQS_RUNTIME_ERROR << "interaction_time must be positive";
 int nb_orders = params.max_perturbation_order - params.min_perturbation_order + 1;
 if (nb_orders < 2) TRIQS_RUNTIME_ERROR << "The range of perturbation orders must cover at least 2 orders";

 // boundaries
 t_max = *std::max_element(params.measure_times.begin(), params.measure_times.end());
 t_max = std::max(t_max, std::get<1>(params.right_input_points[0]));

 // rank of the Green's function to calculate. For now only 1 is supported.
 if (params.right_input_points.size() % 2 == 0) TRIQS_RUNTIME_ERROR << "There must be an odd number of right input points";
 if (params.right_input_points.size() > 1) TRIQS_RUNTIME_ERROR << "For now only rank 1 Green's functions are supported.";
 rank = params.right_input_points.size() / 2;

 op_to_measure_spin = std::get<0>(params.right_input_points[0]); // for now
 taup = make_keldysh_contour_pt(params.right_input_points[0]);

 // make tau list from left input points lists
 shape_tau_array = {params.measure_times.size(), params.measure_keldysh_indices.size()};

 keldysh_contour_pt tau;
 for (auto time : params.measure_times) {
  for (auto k_index : params.measure_keldysh_indices) {
   tau = {params.measure_state, time, k_index};
   tau_list.emplace_back(tau);
  }
 }

 if (tau_list.size() < 1) TRIQS_RUNTIME_ERROR << "No left input point !";

 // prefactor
 prefactor = array<dcomplex, 1>(params.max_perturbation_order - params.min_perturbation_order + 1);
 prefactor() = 1.0;

 if (params.method == 4)
  prefactor() /= 2. * (params.interaction_start + t_max); // only for cofact formula with additional integral

 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 for (int k = 0; k <= params.max_perturbation_order - params.min_perturbation_order; ++k) {
  prefactor(k) *= i_n[(k + params.min_perturbation_order) % 4]; // * i^(k)
 }

 // results arrays
 pn = array<double, 1>(nb_orders);                    // measurement of p_n
 sn = array<dcomplex, 2>(nb_orders, tau_list.size()); // measurement of s_n
 pn_errors = array<double, 1>(nb_orders);
 sn_errors = array<double, 1>(nb_orders);
 pn() = 0;
 sn() = 0;
 pn_errors() = 0;
 sn_errors() = 0;
};

// --------------------------------
void solver_core::set_g0(gf_view<retime, matrix_valued> g0_lesser, gf_view<retime, matrix_valued> g0_greater) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status > not_ready) TRIQS_RUNTIME_ERROR << "Green functions already set up. Cannot change.";
 // non interacting Green function
 green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max};

 // choose measure and weight
 Weight* weight = new two_det_weight(green_function, tau_list[0], taup, op_to_measure_spin);
 Measure* measure = create_measure(params.method, weight);
 integrand = std::make_shared<Integrand>(weight, measure);

 // Register moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 if (params.w_ins_rem > 0) {
  qmc.add_move(moves::insert{integrand, &params, t_max, qmc.get_rng()}, "insertion", params.w_ins_rem);
  qmc.add_move(moves::remove{integrand, &params, t_max, qmc.get_rng()}, "removal", params.w_ins_rem);
 }
 if (params.w_dbl > 0) {
  qmc.add_move(moves::insert2{integrand, &params, t_max, qmc.get_rng()}, "insertion2", params.w_dbl);
  qmc.add_move(moves::remove2{integrand, &params, t_max, qmc.get_rng()}, "removal2", params.w_dbl);
 }
 if (params.w_shift > 0) {
  qmc.add_move(moves::shift{integrand, &params, t_max, qmc.get_rng()}, "shift", params.w_shift);
 }
 if (params.method == 4) {
  qmc.add_move(moves::weight_swap{integrand, &params, t_max, qmc.get_rng()}, "weight swap", params.w_weight_swap);
  qmc.add_move(moves::weight_shift{integrand, &params, t_max, qmc.get_rng()}, "weight shift", params.w_weight_shift);
 }

 qmc.add_measure(Accumulator(integrand, &pn, &sn, &pn_errors, &sn_errors, &nb_measures), "Measurement");

 // order zero values
 g0_values = array<dcomplex, 1>(tau_list.size());
 for (int i = 0; i < tau_list.size(); ++i) {
  g0_values(i) = green_function(tau_list[i], taup);
 }

 status = ready;
};

// --------------------------------
int solver_core::run(const int max_time = -1) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set !";
 // order zero case
 if (params.max_perturbation_order == 0) return order_zero();

 int run_status;

 // warmup if run has not started yet
 if (status == ready) {
  status = running;
  solve_duration = 0;
  if (triqs::mpi::communicator().rank() == 0) std::cout << "Warming up... ";
  run_status = qmc.run(params.n_warmup_cycles, params.length_cycle, triqs::utility::clock_callback(-1), false);
  if (run_status == 2) return finish(run_status); // Received a signal: abort
  if (triqs::mpi::communicator().rank() == 0) std::cout << "done" << std::endl;
  ;
 }

 // accumulate
 if (triqs::mpi::communicator().rank() == 0) std::cout << "Accumulate..." << std::endl << std::endl;
 run_status = qmc.run(params.n_cycles - nb_measures, params.length_cycle, triqs::utility::clock_callback(max_time), true);

 // Collect results
 mpi::communicator world;
 // use self communicator, gathering the data is done in python later
 // mpi::communicator self(mpi::MPI_COMM_SELF); // does not work but I don't know why
 auto self = world.split(world.rank());
 qmc.collect_results(self);

 // prefactor and reshaping sn
 for (int i = 0; i < tau_list.size(); ++i) sn(range(), i) *= prefactor;
 sn_array = reshape_sn(&sn);

 config_list = integrand->weight->config_list;
 config_weight = integrand->weight->config_weight;
 solve_duration = solve_duration + qmc.get_duration();

 // print acceptance rates
 if (triqs::mpi::communicator().rank() == 0) {
  std::cout << "Duration: " << solve_duration << " seconds" << std::endl;
  std::cout << "Progress: " << 100 * nb_measures / params.n_cycles << " %" << std::endl;
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
  std::cout << std::endl;
 }

 if (run_status != 1) return finish(run_status);

 return run_status;
}

// --------------------------------
int solver_core::finish(const int run_status) {
 if (run_status == 1) TRIQS_RUNTIME_ERROR << "finish cannot be called when the run is not finished";

 if (run_status == 0) status = ready;   // finished
 if (run_status == 2) status = aborted; // Received a signal: abort

 if (run_status == 0 and triqs::mpi::communicator().rank() == 0) std::cout << "Run finished" << std::endl << std::endl;
 if (run_status == 2 and triqs::mpi::communicator().rank() == 0) std::cout << "Run aborted" << std::endl << std::endl;

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

int solver_core::order_zero() {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set!";
 if (status > ready) TRIQS_RUNTIME_ERROR << "A run was in progress, order_zero is not allowed";
 if (triqs::mpi::communicator().rank() == 0) std::cout << "Order zero calculation... ";
 // In the order zero case, pn(0) contains c0 and sn(0, :) contains s0 so that c0*s0 = g0 the unperturbed Green's
 // function. Other elements of these arrays are 0.
 pn() = 0;
 sn() = 0;
 pn_errors() = 0;
 sn_errors() = 0;

 if (params.method == 4) {
  // Uses GSL integration (https://www.gnu.org/software/gsl/manual/html_node/Numerical-integration-examples.html)
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
   pn(0) += value;
   pn_errors(0) += value_error;
  }

  gsl_integration_cquad_workspace_free(w);
  sn(0, range()) = g0_values / pn(0);
  // TODO: sn_errors ?
  sn_array = reshape_sn(&sn);

 } else { // singlepoint methods only
  pn(0) = abs(g0_values(0));
  sn(0, 0) = g0_values(0) / pn(0);
  sn_array = reshape_sn(&sn);
 }
 if (triqs::mpi::communicator().rank() == 0) {
  std::cout << "done" << std::endl;
  std::cout << "c0 = " << pn(0) << " error = " << pn_errors(0) << std::endl << std::endl;
 }
 return 0;
}

// --------------------------------
Measure* solver_core::create_measure(const int method, const Weight* weight) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Method used: " << method << std::endl;

 switch (method) {
  // singletime measures
  case 0: // good old way
   if (tau_list.size() > 1) TRIQS_RUNTIME_ERROR << "Trying to use a singlepoint measure with multiple input point";
   return new weight_sign_measure(weight);
   break;
  // case 1: // same as 0 but with different input times
  // return new twodet_single_measure(physics_params);
  // break;
  //// multitimes measures
  // case 2: // same as 1 but multitime
  // return new twodet_multi_measure(physics_params);
  // break;
  case 4: // same as 3 but with addionnal Z0 for the weight left time
          /* going through */
          // case 3: // same as 2 but with cofact formula
   return new twodet_cofact_measure(green_function, &tau_list, taup, op_to_measure_spin, &g0_values);
   break;
  // default
  default:
   TRIQS_RUNTIME_ERROR << "Cannot recognise the method ID";
   break;
 }
}

// -------
array<dcomplex, 3> solver_core::reshape_sn(array<dcomplex, 2>* sn_list) {
 if (second_dim(*sn_list) != tau_list.size()) TRIQS_RUNTIME_ERROR << "The sn list has not the good size to be reshaped.";
 array<dcomplex, 3> sn_array(first_dim(*sn_list), shape_tau_array[0], shape_tau_array[1]);
 int flatten_idx = 0;
 for (int i = 0; i < shape_tau_array[0]; ++i) {
  for (int j = 0; j < shape_tau_array[1]; ++j) {
   sn_array(range(), i, j) = (*sn_list)(range(), flatten_idx);
   flatten_idx++;
  }
 }
 return sn_array;
};
