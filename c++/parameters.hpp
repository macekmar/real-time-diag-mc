#pragma once
#include <vector>
#include <string>
#include <triqs/gfs.hpp>

#define IMPURITY_MATRIX

#ifdef IMPURITY_MATRIX
using x_index_t = int;
#endif

//using namespace triqs::utility;

// All the arguments of the solve function
struct solve_parameters_t {

 /// operator to measure
 std::vector<std::vector<std::tuple<x_index_t, int>>> op_to_measure;

 /// measure times
 std::pair<std::vector<double>, double> measure_times;

 /// fixed weight time
 std::pair<double, double> weight_times;

 /// U
 double U;

 /// Alpha term
 double alpha;

 // weight of insert2 and remove2 compared to insert and remove
 double p_dbl = 0.5;

 // weight of the shift move
 double p_shift = 1.0;

 //weight of the weight_time_swap move
 double p_weight_time_swap = 1.0;

 // ----   QMC parameters

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
 int verbosity = ((triqs::mpi::communicator().rank() == 0) ? 3 : 0); // silence the slave nodes

 /// Method
 int method = 4;
 
};
