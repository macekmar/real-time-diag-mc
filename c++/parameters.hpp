#pragma once
using namespace triqs::utility;

enum spin {up, down};

// All the arguments of the solve function
struct solve_parameters_t {

 /// U
 double U;

 /// width of the lattice
 int L;

 /// tmax
 double tmax;

 /// Alpha term
 double alpha;

 /// probability to jump by 2 orders (insert2 and remove2)
 double p_dbl = 0.5;

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

};
