#include "./solver_core.hpp"
#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace mpi = triqs::mpi;
using triqs::arrays::range;


// ------------ The main class of the solver ------------------------
solver_core::solver_core(solve_parameters_t const& params)
   : qmc(params.random_name, params.random_seed, 1.0, params.verbosity, false), // first_sign is not used
     params(params) {

 if (mpi::communicator().rank() == 0 and params.store_configurations != 0)
  std::cout << "/!\\ Configurations are being stored." << std::endl << std::endl;

 if (params.interaction_start < 0) TRIQS_RUNTIME_ERROR << "interaction_time must be positive";
 if (params.max_perturbation_order - params.min_perturbation_order + 1 < 2)
  TRIQS_RUNTIME_ERROR << "The range of perturbation orders must cover at least 2 orders";

 // Check creation and annihilation operators
 if (params.creation_ops.size() != params.annihilation_ops.size() or
     params.extern_alphas.size() != params.creation_ops.size())
  TRIQS_RUNTIME_ERROR << "Number of creation operators, of annihilation operators and of external alphas must match";

 // Check potential
 auto nb_edges = std::get<0>(params.potential).size();
 auto& i_list = std::get<1>(params.potential);
 auto& j_list = std::get<2>(params.potential);
 if (i_list.size() != nb_edges or j_list.size() != nb_edges)
  TRIQS_RUNTIME_ERROR << "Potential lists have different sizes.";
 if (nb_edges < 1)
  TRIQS_RUNTIME_ERROR << "Potential lists are empty.";
 for (size_t k = 0; k < nb_edges; ++k) {
  if (i_list[k] < 0 or i_list[k] >= params.nb_orbitals or j_list[k] < 0 or j_list[k] >= params.nb_orbitals)
   TRIQS_RUNTIME_ERROR << "Potential lists contain unknown orbitals. Maybe nb_orbitals is too small.";
 }

 for (int rank = 0; rank < params.creation_ops.size(); ++rank) {
  creation_pts.push_back(make_keldysh_contour_pt(params.creation_ops[rank], rank));
  annihila_pts.push_back(make_keldysh_contour_pt(params.annihilation_ops[rank], rank));
 }

 // kernel binning
 kernels_binning =
     KernelBinning(-params.interaction_start, 0., params.nb_bins, params.max_perturbation_order,
                   params.nb_orbitals, false); // TODO: it should be defined in measure

 // results arrays
 pn = array<long, 1>(params.max_perturbation_order + 1);
 pn() = 0;

 // to store old method results
 sn = array<dcomplex, 1>(params.max_perturbation_order + 1);
 sn() = 0;
};

// --------------------------------
// TODO: put the green functions in the parameters dictionnary so as to construct in one step only
void solver_core::set_g0(triqs::gfs::gf_view<triqs::gfs::retime, triqs::gfs::matrix_valued> g0_lesser,
                         triqs::gfs::gf_view<triqs::gfs::retime, triqs::gfs::matrix_valued> g0_greater) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status > not_ready) TRIQS_RUNTIME_ERROR << "Green functions already set up. Cannot change.";

 // orbitals check
 if (second_dim(g0_lesser.data()) != third_dim(g0_lesser.data()))
   TRIQS_RUNTIME_ERROR << "Lesser is not a square matrix !";
 if (second_dim(g0_greater.data()) != third_dim(g0_greater.data()))
   TRIQS_RUNTIME_ERROR << "Greater is not a square matrix !";
 if (second_dim(g0_lesser.data()) != params.nb_orbitals)
   TRIQS_RUNTIME_ERROR << "Lesser matrix size should match the number of orbitals";
 if (second_dim(g0_greater.data()) != params.nb_orbitals)
   TRIQS_RUNTIME_ERROR << "Greater matrix size should match the number of orbitals";

 // non interacting Green function
 green_function = g0_keldysh_t{g0_t{g0_lesser}, g0_t{g0_greater}};
 green_function_alpha = g0_keldysh_alpha_t{green_function, params.alpha, params.extern_alphas};

 // configuration
 config = Configuration(green_function_alpha, annihila_pts, creation_pts, params.max_perturbation_order,
                        params.singular_thresholds, params.method, params.nonfixed_op,
                        params.cycles_trapped_thresh);

 // Register moves and measurements
 if (params.w_ins_rem > 0) {
  qmc.add_move(moves::insert{config, params, qmc.get_rng()}, "insertion", params.w_ins_rem);
  qmc.add_move(moves::remove{config, params, qmc.get_rng()}, "removal", params.w_ins_rem);
 }
 if (params.w_dbl > 0) {
  qmc.add_move(moves::insert2{config, params, qmc.get_rng()}, "insertion2", params.w_dbl);
  qmc.add_move(moves::remove2{config, params, qmc.get_rng()}, "removal2", params.w_dbl);
 }
 if (params.w_shift > 0) {
  qmc.add_move(moves::shift{config, params, qmc.get_rng()}, "shift", params.w_shift);
 }

 if (params.method == 0) {
  qmc.add_measure(WeightSignMeasure(&config, &pn, &sn), "Weight sign measure");
 } else if (params.method == 1 or params.method == 2) {
  qmc.add_measure(TwoDetKernelMeasure(&config, &kernels_binning, &pn, &kernels, &kernel_diracs, &nb_kernels),
                  "Kernel measure");
 } else {
  TRIQS_RUNTIME_ERROR << "Cannot recognise the method ID";
 }

 status = ready;
};

// --------------------------------
// Run the Monte-Carlo, gather results of all world processes and print some info (cumulated over all world processes).
int solver_core::run(const int nb_cycles, const bool do_measure, const int max_time = -1) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set !";
 // order zero case
 if (params.max_perturbation_order == 0)
  TRIQS_RUNTIME_ERROR << "Order zero cannot run";

 mpi::communicator world;
 world.barrier();

 int run_status;

 if (status == ready) {
  status = running;
  qmc_duration = 0;
 }

 // accumulate
 if (world.rank() == 0) std::cout << "Accumulate..." << std::flush;
 run_status = qmc.run(nb_cycles, params.length_cycle, triqs::utility::clock_callback(max_time), do_measure);
 qmc_duration = qmc_duration + qmc.get_duration();
 if (world.rank() == 0) std::cout << "done" << std::endl;

 // Collect results
 if (world.rank() == 0) std::cout << "Collecting results... " << std::flush;
 qmc.collect_results(world);
 cum_qmc_duration = mpi::mpi_all_reduce(qmc_duration);
 if (world.rank() == 0) std::cout << "done" << std::endl;

 array<double, 1> weight_sum = config.get_weight_sum();
 weight_sum = mpi::mpi_all_reduce(weight_sum);
 array<long, 1> nb_values = config.get_nb_values();
 nb_values = mpi::mpi_all_reduce(nb_values);
 array<double, 1> weight_avg = weight_sum / nb_values;

 array<long, 1> nb_cofact = mpi::mpi_all_reduce(config.nb_cofact);
 array<long, 1> nb_inverse = mpi::mpi_all_reduce(config.nb_inverse);

 // print acceptance rates
 if (world.rank() == 0) {
  std::cout << "Duration (all nodes): " << cum_qmc_duration << " seconds" << std::endl;
  std::cout << "Nb of measures (all nodes): " << get_nb_measures() << std::endl;
  std::cout << "Acceptance rates (node 0):" << std::endl;
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
   else
    move_weight = 0;
   if (move_weight > 0) {
    total_weight += move_weight;
    total_rate += move_weight * x.second;
   }

   std::cout << "> " << x.first << " (" << move_weight << "): " << x.second << std::endl;
  }
  std::cout << "> All moves: " << total_rate / total_weight << std::endl;
  std::cout << "Attempted config weight average (all nodes):" << std::endl << weight_avg << std::endl;
  std::cout << "cofact vs inverse (all nodes): " << nb_cofact << " / " << nb_inverse << std::endl;
  std::cout << "regen ratios (node 0): 0=" << config.matrices[0].get_regen_ratio()
            << ", 1=" << config.matrices[1].get_regen_ratio() << std::endl;
  std::pair<double, double> regen_stats;
  for (int a : {0, 1}) {
   regen_stats = config.matrices[a].get_error_stats();
   std::cout << "regen errors (node 0) " << a << " : avg=" << regen_stats.first
             << ", +3std=" << regen_stats.first + 3 * std::sqrt(regen_stats.second)
             << ", +5std=" << regen_stats.first + 5 * std::sqrt(regen_stats.second) << std::endl;
  }
 }

 if (run_status != 1) return finish(run_status);

 return run_status;
}

// --------------------------------
/* Divides world into P = `size_partition` equal parts plus a minimal remain
 * part and collects (cumulates) results within each of these parts.
 *
 * No division is done if `size_partition` is 1, results are cumulated over all
 * processes.
 * Returns true if this process is a master of one of the equal parts, false
 * otherwise.
 *
 * This is usefull for error estimation, as it allows to gather P different but
 * comparable results with a maximal number of process K per result. If N is
 * the total number of processes, this condition is true when N = KP + R with R
 * >= 0 minimal. Therefore K = N // P. The last part, made of R = K % P
 * processes may be ignored.
 *
 * As parts are made of contiguous ranks, color is rank // K = rank // (N // P).
 * The color of the remain part, if it exists, is therefore P.
 *
 * /!\ cumulation may not be revertible ! (FIXME)
 */
bool solver_core::collect_results(int size_partition) {
 mpi::communicator world;

 if (size_partition <= 1) {
  qmc.collect_results(world);
  cum_qmc_duration = mpi::mpi_all_reduce(qmc_duration, world);
  return (world.rank() == 0);
 } else {
  // create partitions
  int nb_parts = std::min(size_partition, world.size());
  int color = world.rank() / (world.size() / nb_parts);
  mpi::communicator part = world.split(color, world.rank());

  // collect within these partitions
  qmc.collect_results(part);
  cum_qmc_duration = mpi::mpi_all_reduce(qmc_duration, part);
  return (part.rank() == 0) and (color < nb_parts);
 }
}

// --------------------------------
int solver_core::finish(const int run_status) {
 if (run_status == 1) TRIQS_RUNTIME_ERROR << "finish cannot be called when the run is not finished";

 if (run_status == 2) status = aborted; // Received a signal: abort

 if (run_status == 0 and mpi::communicator().rank() == 0)
  std::cout << "Run finished" << std::endl;
 if (run_status == 2 and mpi::communicator().rank() == 0)
  std::cout << "Run aborted" << std::endl;

 return run_status;
}

