#include "./solver_core.hpp"
#include "./moves.hpp"
#include <triqs/det_manip.hpp>

using namespace triqs::arrays;
using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using triqs::utility::mindex;
using triqs::arrays::range;


// ------------ The main class of the solver ------------------------
solver_core::solver_core(solve_parameters_t const& params)
   : qmc(params.random_name, params.random_seed, 1.0, params.verbosity, false), // first_sign is not used
     params(params) {

 compilation_time_stamp(48);

#ifdef REGISTER_CONFIG
 if (mpi::communicator().rank() == 0)
  std::cout << "/!\ Configuration Registration ON" << std::endl << std::endl;
#endif

 if (params.interaction_start < 0) TRIQS_RUNTIME_ERROR << "interaction_time must be positive";
 if (params.max_perturbation_order - params.min_perturbation_order + 1 < 2)
  TRIQS_RUNTIME_ERROR << "The range of perturbation orders must cover at least 2 orders";

 // make tau list from first annihilation op points lists
 tau_array = array<keldysh_contour_pt, 2>(params.measure_times.size(), params.measure_keldysh_indices.size());
 if (tau_array.is_empty()) TRIQS_RUNTIME_ERROR << "No left input point !";

 keldysh_contour_pt tau;
 for (int i = 0; i < params.measure_times.size(); ++i) {
  for (int j = 0; j < params.measure_keldysh_indices.size(); ++j) {
   tau = {params.measure_state, params.measure_times[i], params.measure_keldysh_indices[j]};
   tau_array(i, j) = tau;
  }
 }

 // Check creation and annihilation operators
 if (params.creation_ops.size() != params.annihilation_ops.size() + 1)
  TRIQS_RUNTIME_ERROR << "Number of creation operators must match the number of annihilation operators + 1";
 if (params.extern_alphas.size() != params.creation_ops.size())
  TRIQS_RUNTIME_ERROR << "Number of external alphas must match number of creation operators.";

 int rank = 0;
 for (auto const& pt : params.creation_ops) {
  creation_pts.push_back(make_keldysh_contour_pt(pt, rank));
  rank++;
 }

 annihila_pts.push_back(tau_array(0, 0));
 annihila_pts[0].rank = 0;
 rank = 1;
 for (auto const& pt : params.annihilation_ops) {
  annihila_pts.push_back(make_keldysh_contour_pt(pt, rank));
  rank++;
 }

 // boundaries
 t_max = *std::max_element(params.measure_times.begin(), params.measure_times.end());
 for (auto const& pt : creation_pts) {
  if (pt.t > t_max) t_max = pt.t;
 }
 for (auto const& pt : annihila_pts) {
  if (pt.t > t_max) t_max = pt.t;
 }

 // kernel binning
 kernels_all = array<dcomplex, 3>(params.max_perturbation_order, params.nb_bins, 2);
 kernels = array<dcomplex, 3>(params.max_perturbation_order, params.nb_bins, 2);
 kernels_binning =
     KernelBinning(-params.interaction_start, t_max, params.nb_bins, params.max_perturbation_order,
                   false); // TODO: it should be defined in measure

 // results arrays
 pn = array<long, 1>(params.max_perturbation_order + 1);     // pn for this process
 pn_all = array<long, 1>(params.max_perturbation_order + 1); // gather pn for all processes
 pn() = 0;
 pn_all() = 0;

 sn = array<dcomplex, 3>(params.max_perturbation_order + 1, params.measure_times.size(),
                         params.measure_keldysh_indices.size()); // multidim array of sn for this process
 sn_all = array<dcomplex, 3>(params.max_perturbation_order + 1, params.measure_times.size(),
                             params.measure_keldysh_indices.size()); // multidim array of sn for all processes
 sn() = 0;
 sn_all() = 0;
};

// --------------------------------
// TODO: put the green functions in the parameters dictionnary so as to construct in one step only
void solver_core::set_g0(gf_view<retime, matrix_valued> g0_lesser,
                         gf_view<retime, matrix_valued> g0_greater) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status > not_ready) TRIQS_RUNTIME_ERROR << "Green functions already set up. Cannot change.";

 // non interacting Green function
 green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}};
 green_function_alpha = g0_keldysh_alpha_t{green_function, params.alpha, params.extern_alphas};

 // configuration
 bool kernels_method = (params.method != 0);
 config = Configuration(green_function_alpha, annihila_pts, creation_pts, params.max_perturbation_order,
                        params.singular_thresholds, kernels_method);

 // order zero values
 // auto g0 = g0_npart(green_function, annihila_pts, creation_pts); // probably give wrong values
 auto gf_map = map([&](keldysh_contour_pt tau) { return green_function(tau, creation_pts[0]); });
 g0_array = make_matrix(gf_map(tau_array));

 // Register moves and measurements
 if (params.w_ins_rem > 0) {
  qmc.add_move(moves::insert{&config, &params, t_max, qmc.get_rng()}, "insertion", params.w_ins_rem);
  qmc.add_move(moves::remove{&config, &params, t_max, qmc.get_rng()}, "removal", params.w_ins_rem);
 }
 if (params.w_dbl > 0) {
  qmc.add_move(moves::insert2{&config, &params, t_max, qmc.get_rng()}, "insertion2", params.w_dbl);
  qmc.add_move(moves::remove2{&config, &params, t_max, qmc.get_rng()}, "removal2", params.w_dbl);
 }
 if (params.w_shift > 0) {
  qmc.add_move(moves::shift{&config, &params, t_max, qmc.get_rng()}, "shift", params.w_shift);
 }

 if (params.method == 0) {
  if (size(tau_array) > 1)
   TRIQS_RUNTIME_ERROR << "Trying to use a singlepoint measure with multiple input point";
  qmc.add_measure(WeightSignMeasure(&config, &pn, &pn_all, &sn, &sn_all), "Weight sign measure");
 } else if (params.method == 5) {
  qmc.add_measure(TwoDetKernelMeasure(&config, &kernels_binning, &pn, &pn_all, &kernels, &kernels_all),
                  "Kernel measure");
 } else {
  TRIQS_RUNTIME_ERROR << "Cannot recognise the method ID";
 }

 status = ready;
};

// --------------------------------
int solver_core::run(const int nb_cycles, const bool do_measure, const int max_time = -1) {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set !";
 // order zero case
 if (params.max_perturbation_order == 0)
  TRIQS_RUNTIME_ERROR << "Order zero cannot run, use order_zero method instead";

 mpi::communicator world;
 std::cout << "Waiting for other processes before run..." << std::endl;
 MPI_Barrier(MPI_COMM_WORLD);

 int run_status;

 if (status == ready) {
  status = running;
  solve_duration = 0;
 }

 // accumulate
 std::cout << "Accumulate..." << std::endl;
 run_status = qmc.run(nb_cycles, params.length_cycle, triqs::utility::clock_callback(max_time), do_measure);
 std::cout << "done" << std::endl << std::endl;

 // Collect results
 std::cout << "Collecting results... " << std::endl;
 qmc.collect_results(world);
 std::cout << "done" << std::endl << std::endl;

 solve_duration = solve_duration + qmc.get_duration();
 solve_duration_all = mpi::mpi_all_reduce(solve_duration);

 array<double, 1> weight_sum = config.get_weight_sum();
 weight_sum = mpi::mpi_all_reduce(weight_sum);
 array<long, 1> nb_values = config.get_nb_values();
 nb_values = mpi::mpi_all_reduce(nb_values);
 array<double, 1> weight_avg = weight_sum / nb_values;

 array<long, 1> nb_cofact = mpi::mpi_all_reduce(config.nb_cofact);
 array<long, 1> nb_inverse = mpi::mpi_all_reduce(config.nb_inverse);

 // print acceptance rates
 if (world.rank() == 0) {
  std::cout << "Duration: " << solve_duration << " seconds" << std::endl;
  std::cout << "Nb of measures: " << get_nb_measures_all() << std::endl;
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
   else
    move_weight = 0;
   if (move_weight > 0) {
    total_weight += move_weight;
    total_rate += move_weight * x.second;
   }

   std::cout << "> " << x.first << " (" << move_weight << "): " << x.second << std::endl;
  }
  std::cout << "> All moves: " << total_rate / total_weight << std::endl;
  std::cout << "Attempted config weight average:" << std::endl << weight_avg << std::endl;
  std::cout << "cofact vs inverse : " << nb_cofact << " / " << nb_inverse << std::endl;
  std::cout << "regen ratios : 0=" << config.matrices[0].get_regen_ratio()
            << ", 1=" << config.matrices[1].get_regen_ratio() << std::endl;
  std::pair<double, double> regen_stats;
  for (int a : {0, 1}) {
   regen_stats = config.matrices[a].get_error_stats();
   std::cout << "regen errors " << a << " : avg=" << regen_stats.first
             << ", std=" << std::sqrt(regen_stats.second) << std::endl;
  }
  std::cout << std::endl;
 }

 if (run_status != 1) return finish(run_status);

 return run_status;
}

// --------------------------------
void solver_core::compute_sn_from_kernels() {
 // TODO: maybe a more advanced integration (trapeze ?)
 // /!\ Maybe only for one particle GF ?
 auto taup = creation_pts[0];
 if (params.method != 5) TRIQS_RUNTIME_ERROR << "Cannot use kernels with this method";
 std::cout << "Computing sn from kernels..." << std::flush;
 keldysh_contour_pt tau;
 for (int i = 0; i < second_dim(sn); ++i) { // for each tau (time)
  for (int a = 0; a < third_dim(sn); ++a) { // for each tau (keldysh index)
   tau = tau_array(i, a);
   auto gf_map = map([&](keldysh_contour_pt alpha) { return green_function(tau, alpha); });
   auto gf_tau_alpha = gf_map(kernels_binning.coord_array());
   sn(0, i, a) = green_function(tau, taup);
   sn_all(0, i, a) = green_function(tau, taup);
   for (int order = 1; order < first_dim(sn); ++order) { // for each order
    sn(order, i, a) = sum(gf_tau_alpha * kernels(order - 1, ellipsis()));
    sn_all(order, i, a) = sum(gf_tau_alpha * kernels_all(order - 1, ellipsis()));
   }
  }
 }
 std::cout << "done" << std::endl;
}

// --------------------------------
int solver_core::finish(const int run_status) {
 if (run_status == 1) TRIQS_RUNTIME_ERROR << "finish cannot be called when the run is not finished";

 if (run_status == 2) status = aborted; // Received a signal: abort

 if (run_status == 0 and mpi::communicator().rank() == 0)
  std::cout << "Run finished" << std::endl << std::endl;
 if (run_status == 2 and mpi::communicator().rank() == 0)
  std::cout << "Run aborted" << std::endl << std::endl;

 return run_status;
}

// --------------------------------
std::tuple<double, array<dcomplex, 2>> solver_core::order_zero() {
 if (status < not_ready) TRIQS_RUNTIME_ERROR << "Run aborted";
 if (status < ready) TRIQS_RUNTIME_ERROR << "Unperturbed Green's functions have not been set!";
 double c0 = 0;

 if (params.method == 5) {
  c0 = 1.;
 } else { // singlepoint methods only
  c0 = abs(g0_array(0, 0));
 }
 array<dcomplex, 2> s0 = g0_array / c0;
 if (mpi::communicator().rank() == 0) {
  std::cout << "c0 = " << c0 << std::endl << std::endl;
 }
 return std::tuple<double, array<dcomplex, 2>>{c0, s0};
}
