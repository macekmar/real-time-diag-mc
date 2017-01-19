#include "./solver_core.hpp"
#include "./accumulator.hpp"
#include "./measure.hpp"
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

 // Calculate first sign. Needed only for single measurements.
 dcomplex first_sign = 1.0;
 if (physics_params.nb_times == 1) {
  first_sign = physics_params.order_zero_values(0);
  double abs_first_sign = std::abs(first_sign);
  if (abs_first_sign < 1e-12)
   first_sign = 1.0;
  else
   first_sign /= abs_first_sign;
 }

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.random_name, params.random_seed, first_sign, params.verbosity);

 // choose measure, weight and integrand
 two_det_weight weight(&params, &physics_params);

 twodet_cofact_measure measure(&physics_params);
 Measure* measure_p = &measure;
 separated_integrand integrand_mul(&weight, measure_p);
 single_block_integrand integrand_sgl(&weight);
 Integrand* integrand_p = NULL;

 if (physics_params.nb_times > 1) {
  integrand_p = &integrand_mul;
 } else {
  integrand_p = &integrand_sgl;
 }


 // Register moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{integrand_p, &params, &physics_params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{integrand_p, &params, &physics_params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{integrand_p, &params, &physics_params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{integrand_p, &params, &physics_params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_move(moves::shift{integrand_p, &params, &physics_params, qmc.get_rng()}, "shift", params.p_shift);

 if (physics_params.nb_times > 1) // no additional integral if only one time to measure
  qmc.add_move(moves::weight_time_swap{integrand_p, &params, &physics_params, qmc.get_rng()}, "weight time swap",
               params.p_weight_time_swap);

 if (physics_params.nb_times > 1) {
  qmc.add_measure(
      multi_accumulator(static_cast<separated_integrand*>(integrand_p), &pn, &sn, &pn_errors, &sn_errors, &_nb_measures),
      "Multi Measurement");
 } else {
  qmc.add_measure(single_accumulator(integrand_p, &pn, &sn, &pn_errors, &sn_errors, &_nb_measures), "Single Measurement");
 }

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
