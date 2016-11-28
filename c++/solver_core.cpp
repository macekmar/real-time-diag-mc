#include <triqs/det_manip.hpp>
#include <triqs/mc_tools.hpp>
#include "./solver_core.hpp"
#include "./measures.hpp"
#include "./moves.hpp"

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;

// ------------ The main class of the solver ------------------------

// -------------------------------------------------------------------------
// The method that runs the qmc
std::pair<std::pair<array<double, 1>, array<dcomplex, 1>>, std::pair<array<double, 1>, array<double, 1>>>
solver_core::solve(solve_parameters_t const& params) {

 auto pn = array<double, 1>(params.max_perturbation_order + 1); // measurement of p_n
 pn() = 0;
 auto sn = array<dcomplex, 1>(params.max_perturbation_order + 1); // measurement of s_n
 sn() = 0;
 auto pn_errors = pn;
 auto sn_errors = pn;

 // Prepare the data
 auto data = qmc_data_t{params};
 auto t_max = qmc_time_t{data.tmax};

 // Initialize the M-matrices. 100 is the initial matrix size
 for (auto spin : {up, down})
  data.matrices.emplace_back(g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max}, 100);

 for (auto spin : {up, down}) {
  auto const& v = params.op_to_measure[spin];
  if (v.size() == 2) data.matrices[spin].insert_at_end(make_keldysh_contour_pt(v[0]), make_keldysh_contour_pt(v[1]));
 }

if (params.max_perturbation_order == 0) { 
 auto order_zero_value = data.matrices[up].determinant() * data.matrices[down].determinant();
 pn(0) = std::abs(order_zero_value);
 sn(0) = order_zero_value / pn(0) * dcomplex({0, -1}); // FIXME not accurate if pn(0) is very small
 _solve_duration = 0.0;

 return {{pn, sn}, {pn_errors, sn_errors}};
}

 // Compute initial sum of determinants (needed for the first MC move)
 data.sum_keldysh_indices = recompute_sum_keldysh_indices(&data, &params, 0);

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.n_cycles, params.length_cycle, params.n_warmup_cycles,
                                                  params.random_name, params.random_seed, params.verbosity);

 // Register moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{&data, &params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{&data, &params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{&data, &params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{&data, &params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_move(moves::shift{&data, &params, qmc.get_rng()}, "shift", params.p_shift);

 qmc.add_measure(measure_pn_sn{&data, &pn, &sn, &pn_errors, &sn_errors}, "M measurement");

 // Run
 qmc.start(1.0, triqs::utility::clock_callback(params.max_time));

 // Collect results
 mpi::communicator world;
 qmc.collect_results(world);

 _solve_duration = qmc.get_duration();

 return {{pn, sn}, {pn_errors, sn_errors}};
}
