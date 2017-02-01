#include "./solver_core.hpp"
#include "./accumulator.hpp"
#include "./moves.hpp"
#include "./weight.hpp"
#include <triqs/det_manip.hpp>
#include <triqs/mc_tools.hpp>

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;
using triqs::arrays::range;

// ------------ The main class of the solver ------------------------

Measure* solver_core::_create_measure(const int method, const input_physics_data* physics_params, const Weight* weight) {

 switch (method) {
  // singletime measures
  case 0: // good old way
   return new weight_sign_measure(physics_params, weight);
   break;
  case 1: // same as 0 but with different input times
   return new twodet_single_measure(physics_params);
   break;
  // multitimes measures
  case 2: // same as 1 but multitime
   return new twodet_multi_measure(physics_params);
   break;
  case 4: // same as 3 but with addionnal integral for the weight left time
   if (physics_params->nb_times == 1)
    TRIQS_RUNTIME_ERROR << "Trying to use the additionnal integral method with a single input time";
  /* going through */
  case 3: // same as 2 but with cofact formula
   return new twodet_cofact_measure(physics_params);
   break;
  // default
  default:
   return new weight_sign_measure(physics_params, weight);
   break;
 }
}

// -------------------------------------------------------------------------
// The method that runs the qmc
std::pair<std::pair<array<double, 1>, array<dcomplex, 2>>, std::pair<array<double, 1>, array<double, 1>>>
solver_core::solve(solve_parameters_t const& params) {

 // Prepare the data
 input_physics_data physics_params(&params, g0_lesser, g0_greater);

 int nb_orders = params.max_perturbation_order - params.min_perturbation_order + 1;

 auto pn = array<double, 1>(nb_orders);                            // measurement of p_n
 auto sn = array<dcomplex, 2>(nb_orders, physics_params.nb_times); // measurement of s_n
 auto pn_errors = array<double, 1>(nb_orders);
 auto sn_errors = array<double, 1>(nb_orders);
 pn() = 0;
 sn() = 0;
 pn_errors() = 0;
 sn_errors() = 0;

 // order zero case
 if (params.max_perturbation_order == 0) {
  pn() = 1.0;
  sn(0, range()) = physics_params.order_zero_values * dcomplex({0, -1});
  return {{pn, sn}, {pn_errors, sn_errors}};
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
 qmc.collect_results(world);

 // prefactor
 array<dcomplex, 1> prefactor = physics_params.prefactor();
 for (int i = 0; i < physics_params.nb_times; ++i) sn(range(), i) *= prefactor;

 _solve_duration = qmc.get_duration();

 return {{pn, sn}, {pn_errors, sn_errors}};
}
