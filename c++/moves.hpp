#pragma once
#include "./parameters.hpp"
#include "./configuration.hpp"
#include "./qmc_data.hpp"
#include "./random_vertex_gen.hpp"
#include <triqs/mc_tools.hpp>
#include <list>

namespace moves {

std::vector<double> prepare_U(std::vector<double> U);

template <typename Conf>
struct common {
 Conf &config;
 const solve_parameters_t &params;
 const RandomVertexGenerator& rvg;
 triqs::mc_tools::random_generator &rng;
 const std::vector<double> U;
 bool quick_exit = false;

 common(Conf &config, const solve_parameters_t &params,
        triqs::mc_tools::random_generator &rng, const RandomVertexGenerator &rvg)
    : config(config), params(params), rng(rng), rvg(rvg), U(prepare_U(params.U)) {}

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

template <typename Conf>
struct insert : common<Conf> {

 using common<Conf>::common; 
 using common<Conf>::config;
 using common<Conf>::rvg;
 using common<Conf>::rng;
 using common<Conf>::U;
 using common<Conf>::before_attempt;
 using common<Conf>::after_attempt;
 using common<Conf>::is_quick_exit;
 using common<Conf>::quick_exit;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-insertion move --------------------------------------

template <typename Conf>
struct insert2 : common<Conf> {

 using common<Conf>::common; 
 using common<Conf>::config;
 using common<Conf>::rvg;
 using common<Conf>::rng;
 using common<Conf>::U;
 using common<Conf>::before_attempt;
 using common<Conf>::after_attempt;
 using common<Conf>::is_quick_exit;
 using common<Conf>::quick_exit;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC removal move --------------------------------------

template <typename Conf>
struct remove : common<Conf> {
 vertex_t removed_vtx;
 int p;

 using common<Conf>::common; 
 using common<Conf>::config;
 using common<Conf>::rvg;
 using common<Conf>::rng;
 using common<Conf>::U;
 using common<Conf>::before_attempt;
 using common<Conf>::after_attempt;
 using common<Conf>::is_quick_exit;
 using common<Conf>::quick_exit;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-removal move --------------------------------------

template <typename Conf>
struct remove2 : common<Conf> {
 vertex_t removed_vtx1, removed_vtx2;
 int p1, p2;

 using common<Conf>::common; 
 using common<Conf>::config;
 using common<Conf>::rvg;
 using common<Conf>::rng;
 using common<Conf>::U;
 using common<Conf>::before_attempt;
 using common<Conf>::after_attempt;
 using common<Conf>::is_quick_exit;
 using common<Conf>::quick_exit;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

//-----------QMC vertex shift move------------

template <typename Conf>
struct shift : common<Conf> {
 vertex_t removed_vtx;
 int p;

 using common<Conf>::common; 
 using common<Conf>::config;
 using common<Conf>::rvg;
 using common<Conf>::rng;
 using common<Conf>::U;
 using common<Conf>::before_attempt;
 using common<Conf>::after_attempt;
 using common<Conf>::is_quick_exit;
 using common<Conf>::quick_exit;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC Auxillary MC move --------------------------------------


struct auxmc : public common<ConfigurationQMC> {
 using common<ConfigurationQMC>::common;
using common<ConfigurationQMC>::config;
 using common<ConfigurationQMC>::rvg;
 using common<ConfigurationQMC>::U;
 using common<ConfigurationQMC>::before_attempt;
 using common<ConfigurationQMC>::after_attempt;
 using common<ConfigurationQMC>::is_quick_exit;
 using common<ConfigurationQMC>::quick_exit;
  
 bool move_accepted = false; 
//  ConfigurationAuxMC* aux_config;
 ConfigurationAuxMC* aux_config;
 triqs::mc_tools::mc_generic<dcomplex>* aux_mc;

 std::list<vertex_t> vertices;
 std::list<vertex_t> old_vertices;

 dcomplex attempt();
 dcomplex accept();
 void reject();
};

}


