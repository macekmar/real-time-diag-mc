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

 /// input contour points, except the first (left) one, must be of odd size
 std::vector<std::tuple<x_index_t, double, int>> right_input_points;

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

 /// weight of the weight_swap move
 double w_weight_swap = 0.01;

 /// weight of the weight_shift move
 double w_weight_shift = 0.01;

 /// Maximum order in U
 int max_perturbation_order = 3;

 /// Minimal order in U
 int min_perturbation_order = 0;

 /// Number of QMC cycles
 int n_cycles;

 /// Length of a single QMC cycle
 int length_cycle = 50;

 /// Number of cycles for thermalization
 int n_warmup_cycles = 5000;

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
 int method = 4;

 /// nb of bins for the kernels
 int nb_bins = 10000;

 /// weight minimum value
 // TODO: rename in weight_offset or something
 double weight_min = 0.;
 
};
