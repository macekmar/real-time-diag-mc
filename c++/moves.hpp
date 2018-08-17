#pragma once
#include "./measure.hpp"
#include "./parameters.hpp"
#include "./qmc_data.hpp"

namespace moves {

struct common {
 Configuration &config;
 const solve_parameters_t &params;
 vertex_rand_gen vrg;
 triqs::mc_tools::random_generator &rng;
 double normalization;
 bool quick_exit = false;

 common(Configuration &config, const solve_parameters_t &params, const double t_max,
        triqs::mc_tools::random_generator &rng)
    : config(config),
      params(params),
      vrg{{params.nb_orbitals, std::get<0>(params.potential), std::get<1>(params.potential),
           std::get<2>(params.potential)}, params.interaction_start, rng},
      rng(rng) {
  normalization = vrg.size() * params.U;
 }

 /// Tell if `k` is an allowed order
 bool is_quick_exit(int const &k) {
  return k < params.min_perturbation_order or params.max_perturbation_order < k or
         (params.forbid_parity_order != -1 and k % 2 == params.forbid_parity_order and
          k != params.min_perturbation_order);
 }

 /// Construct random vertex
 inline vertex_t get_random_vertex() { return vrg(); }
};

// ------------ QMC insertion move --------------------------------------

struct insert : common {

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-insertion move --------------------------------------

struct insert2 : common {

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC removal move --------------------------------------

struct remove : common {
 vertex_t removed_vtx;
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-removal move --------------------------------------

struct remove2 : common {
 vertex_t removed_vtx1, removed_vtx2;
 int p1, p2;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

//-----------QMC vertex shift move------------

struct shift : common {
 vertex_t removed_vtx;
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};
}
