#include "./solver_core.hpp"
#include "./accumulator.hpp"
#include "./moves.hpp"
#include "./weight.hpp"
#include <gsl/gsl_integration.h>
#include <triqs/det_manip.hpp>
#include <triqs/mc_tools.hpp>

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;
using triqs::arrays::range;

struct integrand_params {
 g0_keldysh_t green_function;
 keldysh_contour_pt tau;
 keldysh_contour_pt taup;

 integrand_params(input_physics_data* physics_params, int state, int k_index)
    : green_function(physics_params->green_function), tau{state, 0., k_index}, taup(physics_params->taup){};
};

double abs_g0_keldysh_t_inputs(double t, void* _params) {
 auto params = static_cast<integrand_params*>(_params);
 keldysh_contour_pt tau = params->tau;
 tau.t = t;
 return std::abs(params->green_function(tau, params->taup));
}
// ------------ The main class of the solver ------------------------

Measure* solver_core::_create_measure(const int method, const input_physics_data* physics_params, const Weight* weight) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Method used: " << method << std::endl;

 switch (method) {
  // singletime measures
  case 0: // good old way
   return new weight_sign_measure(physics_params, weight);
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
   return new twodet_cofact_measure(physics_params);
   break;
  // default
  default:
   TRIQS_RUNTIME_ERROR << "Cannot recognise the method ID";
   break;
 }
}

// -------------------------------------------------------------------------
// The method that runs the qmc
std::pair<std::pair<array<double, 1>, array<dcomplex, 3>>, std::pair<array<double, 1>, array<double, 1>>>
solver_core::solve(solve_parameters_t const& params) {

 // Prepare the data
 input_physics_data physics_params(&params, g0_lesser, g0_greater);

 int nb_orders = params.max_perturbation_order - params.min_perturbation_order + 1;

 auto pn = array<double, 1>(nb_orders);                                   // measurement of p_n
 auto sn = array<dcomplex, 2>(nb_orders, physics_params.tau_list.size()); // measurement of s_n
 auto pn_errors = array<double, 1>(nb_orders);
 auto sn_errors = array<double, 1>(nb_orders);
 pn() = 0;
 sn() = 0;
 pn_errors() = 0;
 sn_errors() = 0;

 // Order zero case
 if (params.max_perturbation_order == 0) {
  // In the order zero case, pn contains c0 and sn contains s0 so that c0*s0 = g0 the unperturbed Green's
  // function.

  if (params.method == 4) {
   // Uses GSL integration (https://www.gnu.org/software/gsl/manual/html_node/Numerical-integration-examples.html)
   gsl_function F;
   F.function = &abs_g0_keldysh_t_inputs;
   auto w = gsl_integration_cquad_workspace_alloc(10000);
   size_t nb_evals;
   double value;
   double value_error;

   for (int a : {0, 1}) {
    integrand_params int_params(&physics_params, 0, a);
    F.params = &int_params;
    gsl_integration_cquad(&F,                                // function to integrate
                          -physics_params.interaction_start, // lower boundary
                          physics_params.t_max,              // upper boundary
                          1e-6,                              // absolute error
                          1e-6,                              // relative error
                          w,                                 // workspace
                          &value,                            // result
                          &value_error,                      // output absolute error
                          &nb_evals);                        // number of function evaluation
    pn(0) += value;
    pn_errors(0) += value_error;
   }

   gsl_integration_cquad_workspace_free(w);
   sn(0, range()) = physics_params.g0_values / pn(0);
   // TODO: sn_errors ?
   auto sn_array = physics_params.reshape_sn(&sn);
   return {{pn, sn_array}, {pn_errors, sn_errors}};

  } else { // singlepoint methods only
   pn(0) = abs(physics_params.g0_values(0));
   sn(0, 0) = physics_params.g0_values(0) / pn(0);
   auto sn_array = physics_params.reshape_sn(&sn);
   return {{pn, sn_array}, {pn_errors, sn_errors}};
  }
 }

 // Construct a Monte Carlo loop (first sign is not used)
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.random_name, params.random_seed, 1.0, params.verbosity);

 // choose measure and weight
 two_det_weight weight(&params, &physics_params);
 Measure* measure_p = _create_measure(params.method, &physics_params, &weight);
 Integrand integrand(&weight, measure_p);

 // Register moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{&integrand, &params, &physics_params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{&integrand, &params, &physics_params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{&integrand, &params, &physics_params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{&integrand, &params, &physics_params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_move(moves::shift{&integrand, &params, &physics_params, qmc.get_rng()}, "shift", params.p_shift);

 if (params.method == 4)
  qmc.add_move(moves::weight_time_swap{&integrand, &params, &physics_params, qmc.get_rng()}, "weight time swap",
               params.p_weight_time_swap);

 qmc.add_measure(Accumulator(&integrand, &pn, &sn, &pn_errors, &sn_errors, &_nb_measures), "Measurement");

 // Run
 qmc.warmup_and_accumulate(params.n_warmup_cycles, params.n_cycles, params.length_cycle,
                           triqs::utility::clock_callback(params.max_time));

 // Collect results
 mpi::communicator world;
 // use self communicator, gathering the data is done in python later
 // mpi::communicator self(mpi::MPI_COMM_SELF); // does not work but I don't know why
 auto self = world.split(world.rank());
 qmc.collect_results(self);

 // prefactor and reshaping sn
 array<dcomplex, 1> prefactor = physics_params.prefactor();
 for (int i = 0; i < physics_params.tau_list.size(); ++i) sn(range(), i) *= prefactor;
 auto sn_array = physics_params.reshape_sn(&sn);

 _solve_duration = qmc.get_duration();

 return {{pn, sn_array}, {pn_errors, sn_errors}};
}
