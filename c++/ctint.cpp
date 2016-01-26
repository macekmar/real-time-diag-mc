#include "ctint.hpp"
#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <triqs/gfs.hpp>
#include <triqs/gfs/bz.hpp>
#include <triqs/gfs/cyclic_lattice.hpp>
#include "./qmc_data.hpp"
#include "./moves.hpp"
#include "./measures.hpp"
#include "./g0_latt.hpp"

using namespace triqs::arrays;
using namespace triqs::lattice;
using namespace triqs::gfs;
using namespace triqs::arrays;
using triqs::utility::mindex;

// ------------ The main class of the solver ------------------------

ctint_solver::ctint_solver(double beta, double mu, int n_freq, double t_min, double t_max, int L, int Lk) {
 std::tie(g0_lesser, g0_greater) = make_g0_lattice(beta, mu, n_freq, t_min, t_max, L, Lk);
}

// -------------------------------------------------------------------------
// The method that runs the qmc
std::pair<array<double, 1>, array<double, 1>> ctint_solver::solve(solve_parameters_t const &params) {

 auto cn = array<double, 1>(params.max_perturbation_order + 1); ///< measurement of c_n
 cn() = 0;
 auto sn = cn;

 // Prepare the data
 auto data = qmc_data_t{};
 auto t_max = qmc_time_t{params.tmax};

 // Initialize the M-matrices. 100 is the initial matrix size
 for (auto spin : {up, down}) data.matrices.emplace_back(g0_keldysh_t{g0_lesser, g0_greater, params.alpha, t_max}, 100);

 // Insert the operators to be measured.
 // We measure the density
 // For up, we insert the fixed pair of times (t_max, t_max), Keldysh index +-.
 //FIXME: Code dependent
 data.matrices[up].insert_at_end({mindex(0, 0, 0), t_max, 0}, {mindex(0, 0, 0), t_max, 1}); // C^+ C

 cn(0)= imag(data.matrices[up].determinant() * data.matrices[down].determinant());
 sn(0)= 1;
 if (params.max_perturbation_order == 0) return {cn, sn};

 // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.n_cycles, params.length_cycle, params.n_warmup_cycles,
                                                  params.random_name, params.random_seed, params.verbosity);

 // Register moves and measurements
 if (params.p_dbl < 1) {
  qmc.add_move(moves::insert{&data, &params, qmc.get_rng()}, "insertion", 1. - params.p_dbl);
  qmc.add_move(moves::remove{&data, &params, qmc.get_rng()}, "removal", 1. - params.p_dbl);
 }
 if (params.p_dbl > 0) {
  qmc.add_move(moves::insert2{&data, &params, qmc.get_rng()}, "insertion2", params.p_dbl);
  qmc.add_move(moves::remove2{&data, &params, qmc.get_rng()}, "removal2", params.p_dbl);
 }
 qmc.add_measure(measure_cs{&data, &cn, &sn}, "M measurement");

 // Run
 qmc.start(1.0, triqs::utility::clock_callback(params.max_time));
 
 // Collect results
 mpi::communicator world;
 qmc.collect_results(world);
 return {cn, sn};

}

