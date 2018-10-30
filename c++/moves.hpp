#pragma once
#include "./measure.hpp"
#include "./parameters.hpp"
#include "./qmc_data.hpp"
#include "./random_vertex_gen.hpp"
#include <triqs/mc_tools.hpp>

namespace moves {


struct common {
 Configuration &config;
 const solve_parameters_t &params;
 const RandomVertexGenerator& rvg;
 triqs::mc_tools::random_generator &rng;
 const double normalization;
 bool quick_exit = false;

 common(Configuration &config, const solve_parameters_t &params,
        triqs::mc_tools::random_generator &rng, const RandomVertexGenerator &rvg)
    : config(config), params(params), rng(rng), rvg(rvg), normalization(params.U) {}

 /// Tell if `k` is an allowed order
 bool is_quick_exit(int const &k) {
  return k < params.min_perturbation_order or params.max_perturbation_order < k or
         (params.forbid_parity_order != -1 and k % 2 == params.forbid_parity_order and
          k != params.min_perturbation_order);
 }

 // things to do before any attempt
 // has to be called at beginning of all moves attempt method
 inline void before_attempt() {
  if (params.store_configurations == 1) config.register_accepted_config();
 };

 // things to do after any attempt (ie before any accept or reject)
 // has to be called at end of all moves attempt method
 inline void after_attempt() {
  if (params.store_configurations == 2) config.register_attempted_config();
 };
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
