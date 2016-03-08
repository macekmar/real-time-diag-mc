#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include "./moves.hpp"
#include "./measures.hpp"
#include "./solver_core.hpp"

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;

// ------------ The main class of the solver ------------------------

// -------------------------------------------------------------------------
// The method that runs the qmc
std::pair<std::pair<array<double, 1>, array<double, 1>>, std::pair<array<double, 1>, array<double, 1>>>
solver_core::solve(solve_parameters_t const &params) {

 auto pn = array<double, 1>(params.max_perturbation_order + 1); // measurement of p_n
 pn() = 0;
 auto sn = pn;
 auto pn_errors = pn;
 auto sn_errors = sn;

 // Prepare the data
 auto data = qmc_data_t{};
 auto t_max = qmc_time_t{params.tmax};

 // Initialize the M-matrices. 100 is the initial matrix size
 for (auto spin : {up, down})
  data.matrices.emplace_back(g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max}, 100);

 // Insert the operators to be measured
 // Set pn(0) and sn(0)
 if (params.measure == "n") { // Measure the density d_up^+ d_up
  // For up, we insert the fixed pair of times (t_max, t_max), Keldysh index +-.
  data.matrices[up].insert_at_end({x_index_t{}, t_max, 0}, {x_index_t{}, t_max, 1});
  pn(0) = imag(data.matrices[up].determinant() * data.matrices[down].determinant());
 } else if (params.measure == "nn") { // Measure the double occupation d_up^+ d_up d_dn^+ d_dn, factor of -i in keldysh sum
  data.matrices[up].insert_at_end({x_index_t{}, t_max, 0}, {x_index_t{}, t_max, 1});
  data.matrices[down].insert_at_end({x_index_t{}, t_max, 0}, {x_index_t{}, t_max, 1});
  pn(0) = -real(data.matrices[up].determinant() * data.matrices[down].determinant());
 } else if (params.measure == "I") { // Measure the current
  data.matrices[up].insert_at_end({x_index_t{0}, t_max, 0}, {x_index_t{1}, t_max, 1});
  pn(0) = 2. * real(data.matrices[up].determinant() * data.matrices[down].determinant());
 } else {
  TRIQS_RUNTIME_ERROR << "Measure '" << params.measure << "' not recognised.";
 }

 if (params.max_perturbation_order == 0) return {{pn, {1}}, {pn_errors, sn_errors}};

 // Compute initial sum of determinants
 data.sum_keldysh_indices = recompute_sum_keldysh_indices(&data, &params, 0);

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.n_cycles, params.length_cycle, params.n_warmup_cycles,
                                                  params.random_name, params.random_seed, params.verbosity);

 // Regeister moves and measurements
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
