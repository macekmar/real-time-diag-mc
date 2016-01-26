#pragma once
#include <triqs/mc_tools.hpp>
#include "./configuration.hpp"

using namespace triqs::gfs; namespace mpi=triqs::mpi;

// ------------ QMC insertion move --------------------------------------

struct move_insert {
 configuration *config;
 const solve_parameters_t *params;
 triqs::mc_tools::random_generator &rng;
 bool quick_exit = false;
 dcomplex sum_dets = 0;

 move_insert(configuration *config, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
    : config(config), params(params), rng(rng) {}
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-insertion move --------------------------------------

struct move_insert2 {
  configuration *config;
  const solve_parameters_t *params;
  triqs::mc_tools::random_generator &rng;
  bool quick_exit = false;
  dcomplex sum_dets = 0;

 move_insert2(configuration *config, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
    : config(config), params(params), rng(rng) {}
  dcomplex attempt();
  dcomplex accept();
  void reject();
};

// ------------ QMC removal move --------------------------------------

struct move_remove {
 configuration *config;
 const solve_parameters_t *params;
 triqs::mc_tools::random_generator &rng;
 point removed_pt;
 bool quick_exit = false;
 dcomplex sum_dets = 0;

 move_remove(configuration *config, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
    : config(config), params(params), rng(rng) {}
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-removal move --------------------------------------

struct move_remove2 {
  configuration *config;
  const solve_parameters_t *params;
  triqs::mc_tools::random_generator &rng;
  point removed_pt1, removed_pt2;
  bool quick_exit = false;
  dcomplex sum_dets = 0;

  move_remove2(configuration *config, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
  : config(config), params(params), rng(rng) {}
  dcomplex attempt();
  dcomplex accept();
  void reject();
};

//  -------------- QMC measurement ----------------
struct measure_cs {

 configuration const *config; // Pointer to the MC configuration
 array<double, 2> &cn_sn;
 long Z = 0;

 measure_cs(configuration const &config_, array<double, 2> &cn_sn) : config(&config_), cn_sn(cn_sn) { cn_sn() = 0; }

 void accumulate(dcomplex sign) {
  Z++;
  int N = config->perturbation_order();
  cn_sn(0, N) += 1;
  cn_sn(1, N) += real(sign);
 }

 void collect_results(mpi::communicator const &c) {
  Z = mpi::all_reduce(Z);
  mpi::all_reduce_in_place(cn_sn);
  for (int i = 0; i < second_dim(cn_sn()); ++i) {
   if (std::isnormal(cn_sn(0, i))) cn_sn(1, i) /= cn_sn(0, i);
   cn_sn(0, i) /= Z;
  }
 }
};

