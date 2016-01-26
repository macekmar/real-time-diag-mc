#include "ctint.hpp"
#include <boost/mpi/environment.hpp>


// short code for profiling
int main(int argc, char* argv[]) {

  // Initialize mpi
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  // constructor parameters
  double U = 6.0;
  double U_qmc = 2.0;
  double beta = 4.0;
  double alpha = 0.4;
  double mu = 0.5*U - U*alpha;
  long n_freq = 500;
  double t_min = -30.0;
  double t_max =  30.0;
  long L = 30;
  long Lk = 100;
  long Nmax = 8;

  // Construct CTQMC solver
  auto ctqmc = ctint_solver(beta, mu, n_freq, t_min, t_max, L, Lk);

  // MC parameters
  solve_parameters_t params;
  params.U = U_qmc;
  params.L = L;
  params.tmax = 10.0;
  params.alpha = alpha;
  params.n_cycles = 50000;
  params.n_warmup_cycles = 100;
  params.length_cycle = 20;
  params.p_dbl = -0.5;
  params.max_perturbation_order = Nmax;

  // Solve!
  ctqmc.solve(params);

  return 0;

}
