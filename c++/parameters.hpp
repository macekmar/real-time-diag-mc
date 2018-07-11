#pragma once
#include <vector>
#include <string>

//template <typename T> using view_t = typename T::view_type;
using namespace triqs::gfs;
//using my_gf_view = gf<retime, matrix_valued>::view_type;

#define IMPURITY_MATRIX

#ifdef IMPURITY_MATRIX
using x_index_t = int;
#endif

//using namespace triqs::utility;

// All the arguments of the solve function
struct solve_parameters_t {

 /// Keldysh contour points for the creation operators
 std::vector<std::tuple<x_index_t, double, int>> creation_ops;

 /// Keldysh contour points for the annihilation operators, except the first one.
 // Its size must match `creation_ops` size - 1
 std::vector<std::tuple<x_index_t, double, int>> annihilation_ops;

 /// alphas to be used in the extra lines/columns. Same length as `creation_ops`.
 std::vector<dcomplex> extern_alphas;

 /// time before 0 at which interaction started
 double interaction_start;

 /// measure states (for the first input point), for now just one
 int measure_state = 0;

 /// measure times (for the first input point)
 std::vector<double> measure_times;

 /// measure keldysh indices (for the first input point)
 std::vector<int> measure_keldysh_indices;

 /// Alpha term
 double alpha;

 // ----   QMC parameters

 /// U
 double U;

 /// weight of insert and remove
 double w_ins_rem = 1.0;

 /// weight of insert2 and remove2
 double w_dbl = 0.5;

 /// weight of the shift move
 double w_shift = 0.0;

 /// Maximum order in U
 int max_perturbation_order = 3;

 /// Minimal order in U
 int min_perturbation_order = 0;

 /// Parity of the orders automatically rejected. -1 (default) to reject no order.
 // In any case min_perturbation_order is never rejected.
 int forbid_parity_order = -1;

 /// Length of a single QMC cycle
 int length_cycle = 50;

 /// Seed for random number generator
 int random_seed = 34788 + 928374 * triqs::mpi::communicator().rank();

 /// Name of random number generator
 std::string random_name = "";

 /// Maximum runtime in seconds, use -1 to set infinite
 int max_time = -1;

 /// Verbosity level
 int verbosity = 0; // silence triqs qmc display
 //int verbosity = ((triqs::mpi::communicator().rank() == 0) ? 3 : 0); // silence the slave nodes

 /// Method
 int method = 5;

 /// nb of bins for the kernels
 int nb_bins = 10000;

 /// log10 conditioning thresholds for each det_manip objects
 std::pair<double, double> singular_thresholds;

 /// Number of cycles after which a trapped configuration is reevaluated
 int cycles_trapped_thresh = 100;

 /// Store the list of all configurations accepted (if 1) or attempted (if 2) in the Markov chain.
 //  Disable the feature if 0.
 //  /!\ will use huge amount of memory, use with small number of cycles only !
 int store_configurations = 0;

};
