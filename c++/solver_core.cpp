#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <triqs/statistics.hpp>
#include "./moves.hpp"
#include "./measures.hpp"
#include "./solver_core.hpp"

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;
using namespace triqs::statistics;

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
 auto observable_pn = array<observable<double> ,1>(params.max_perturbation_order + 1);
 auto observable_sn = array<observable<double> ,1>(params.max_perturbation_order + 1);

 // Prepare the data
 auto data = qmc_data_t{};
 auto t_max = qmc_time_t{params.tmax};

 // FIXME
 data.g0_lesser_0 = g0_lesser(0)(0, 0);

 // Initialize the M-matrices. 100 is the initial matrix size //FIXME why is this size hardcoded?
 for (auto spin : {up, down})
  data.matrices.emplace_back(g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params.alpha, t_max}, 100);

 // Insert the operators to be measured.
 // We measure the density
 // For up, we insert the fixed pair of times (t_max, t_max), Keldysh index +-.
 // FIXME: Code dependent
 data.matrices[up].insert_at_end({x_index_t{}, t_max, 0}, {x_index_t{}, t_max, 1}); // C^+ C
 
 //For the double density (do not forget that there is also a factor of -i in the keldysh_sum.hpp)
// data.matrices[down].insert_at_end({x_index_t{}, t_max, 0}, {x_index_t{}, t_max, 1}); 

 //For the double density FIXME hardcoded
 //pn(0) = - real(data.matrices[up].determinant() * data.matrices[down].determinant());
 //sn(0) = 1;
 
 if (params.max_perturbation_order == 0)
  return {{{imag(data.matrices[up].determinant() * data.matrices[down].determinant())}, {1}}, {pn_errors, sn_errors}};

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.n_cycles, params.length_cycle, params.n_warmup_cycles,
                                                  params.random_name, params.random_seed, params.verbosity);

 // Regeister moves and measurements
 // Can add single moves only, or double moves only (for the case with ph symmetry), or both simultaneously
 // FIXME change to pointers
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{&data, &params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{&data, &params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{&data, &params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{&data, &params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_move(moves::shift{&data, &params, qmc.get_rng()}, "shift", params.p_shift);


 qmc.add_measure(measure_pn_sn{&data, &observable_pn, &observable_sn}, "M measurement");

 // Run
 qmc.start(1.0, triqs::utility::clock_callback(params.max_time));

 // Collect results
 mpi::communicator world;
 qmc.collect_results(world);


 //Now we treat the data to get the correct average values and errors.
 for (int i = 0; i <= params.max_perturbation_order;i++) 
 {
    auto aver_and_err_pn =average_and_error(observable_pn(i));
    auto aver_and_err_sn =average_and_error(observable_sn(i));
    pn(i) = aver_and_err_pn.value;
    sn(i) = aver_and_err_sn.value;
    pn_errors(i) = aver_and_err_pn.error_bar;
    sn_errors(i) = aver_and_err_sn.error_bar;
 }

 
 return {{pn, sn}, {pn_errors, sn_errors}};
}
