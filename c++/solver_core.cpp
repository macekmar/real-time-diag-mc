#include "./solver_core.hpp"
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

 int nb_orders = params.max_perturbation_order - params.min_perturbation_order + 1;

 auto pn = array<double, 1>(nb_orders);                                      // measurement of p_n
 auto sn = array<dcomplex, 2>(nb_orders, params.measure_times.first.size()); // measurement of s_n
 auto pn_errors = array<double, 1>(nb_orders);
 auto sn_errors = array<double, 1>(nb_orders);
 pn() = 0;
 sn() = 0;
 pn_errors() = 0;
 sn_errors() = 0;

 // Prepare the data
 auto t_max = *std::max_element(params.measure_times.first.begin(), params.measure_times.first.end());
 t_max = std::max(t_max, params.measure_times.second);
 int nb_operators = params.op_to_measure[up].size() + params.op_to_measure[down].size();

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.random_name, params.random_seed, 1.0, params.verbosity);

 // non interacting Green function
 auto green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max};

 qmc_weight weight(&params, &green_function);
 qmc_measure measure(&params, green_function);

 // Compute initial sum of determinants (needed for the first MC move)
 measure.evaluate();

 if (params.max_perturbation_order == 0) {
  pn(0) = 1;
  sn(0, range()) = measure.get_value() * dcomplex({0, -1});
  _solve_duration = 0.0;

  return {{pn, sn}, {pn_errors, sn_errors}};
 }

 // Register moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{&measure, &weight, t_max, &params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{&measure, &weight, t_max, &params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{&measure, &weight, t_max, &params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{&measure, &weight, t_max, &params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_move(moves::shift{&measure, &weight, t_max, &params, qmc.get_rng()}, "shift", params.p_shift);
 if (params.measure_times.first.size() > 1) // no additional integral if only one time to measure
  qmc.add_move(moves::weight_time_swap{&measure, &weight, t_max, &params, qmc.get_rng()}, "weight time swap",
               params.p_weight_time_swap);

 qmc.add_measure(qmc_accumulator(&measure, &weight, &pn, &sn, &pn_errors, &sn_errors, &_nb_measures), "M measurement");

 // Run
 qmc.warmup_and_accumulate(params.n_warmup_cycles, params.n_cycles, params.length_cycle,
                           triqs::utility::clock_callback(params.max_time));

 // Collect results
 mpi::communicator world;
 qmc.collect_results(world);

 // Prefactor
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

 for (int k = 0; k <= params.max_perturbation_order; ++k) {
  sn(k, range()) *= i_n[k % 4]; // * i^(k)
  if (nb_operators == 2)
   sn(k, range()) *= dcomplex({0, -1}); // additional factor of -i
  else if (nb_operators == 4)
   sn(k, range()) *= i_n[2]; // additional factor of -1=i^6
  else
   TRIQS_RUNTIME_ERROR << "Operator to measure not recognised.";
 }

 if (params.measure_times.first.size() > 1)
  sn() /= (weight.t_left_max - weight.t_left_min);

      _solve_duration = qmc.get_duration();

 return {{pn, sn}, {pn_errors, sn_errors}};
}
