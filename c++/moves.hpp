#pragma once
#include "./measure.hpp"
#include "./parameters.hpp"
#include "./qmc_data.hpp"

namespace moves {

/// Old random vertex generator
struct old_rand_vertex_gen{
 potential_data_t potential_data;
 timec_t t_max;
 triqs::mc_tools::random_generator &rng;

 /// return a random vertex
 vertex_t operator()() {
  int k = rng(potential_data.values.size()); // random orbital
  return {potential_data.i_list[k], potential_data.j_list[k], -rng(t_max), 0, potential_data.values[k]};
 };

 /// return the probability to have chosen vtx in the *current* configuration
 double probability(vertex_t vtx) {
  return 1 / (potential_data.values.size() * t_max);
 };
};

/// Vertex random generator
class rand_vertex_gen{
 potential_data_t potential_data;
 const timec_t t_max;
 triqs::mc_tools::random_generator &rng;
 Configuration &config;
 const double gamma;
 const double gamma_sqr;

 /// sampling distribution with max = 1
 double pref_sampl_distrib(timec_t t) {
  auto it = config.times_list.upper_bound(t);

  /// find nearest time
  timec_t t0;
  if (it == config.times_list.begin())
   t0 = *it;
  else if (it == config.times_list.end())
   t0 = *--it;
  else if (std::abs(t - *it) < std::abs(t - *std::prev(it)))
   t0 = *it;
  else
   t0 = *--it;

  return gamma_sqr / ((t - t0) * (t - t0) + gamma_sqr);
 };

 /// return integral (between -t_max and 0) of the sampling distribution
 double distrib_norm() {
  double norm = 0;
  double lower = -t_max;
  double upper = -t_max;
  for (auto it = config.times_list.begin(); it != config.times_list.end(); ++it) {
   lower = upper;
   if (std::next(it) == config.times_list.end())
    upper = 0.0;
   else
    upper = 0.5 * (*it + *std::next(it));
   norm += std::atan((upper - *it) / gamma) - std::atan((lower - *it) / gamma);
  }
  norm *= gamma;
  return norm;
 };

 /// generates a random time according to the sampling distribution
 timec_t random_time_generator() {
  timec_t time = -rng(t_max);
  double proba = pref_sampl_distrib(time);
  while (rng(1.0) > proba) {
   time = -rng(t_max);
   proba = pref_sampl_distrib(time);
  }
  return time;
 };

 public:
 rand_vertex_gen(potential_data_t potential_data, timec_t t_max, triqs::mc_tools::random_generator &rng, Configuration &config, double gamma = 1)
: potential_data(potential_data), t_max(t_max), rng(rng), config(config), gamma(gamma), gamma_sqr(gamma*gamma) {};

 /// return a random vertex
 vertex_t operator()() {
  int k = rng(potential_data.values.size()); // random orbital pair
  return {potential_data.i_list[k], potential_data.j_list[k], random_time_generator(), 0, potential_data.values[k]};
 };

 /// return the probability to have chosen vtx in the *current* configuration
 double probability(vertex_t vtx) {
  return pref_sampl_distrib(vtx.t) / (distrib_norm() * potential_data.values.size());
 };
};


struct common {
 Configuration &config;
 const solve_parameters_t &params;
 rand_vertex_gen rvg;
 triqs::mc_tools::random_generator &rng;
 double normalization;
 bool quick_exit = false;

 common(Configuration &config, const solve_parameters_t &params,
        triqs::mc_tools::random_generator &rng)
    : config(config),
      params(params),
      rvg{{params.nb_orbitals, std::get<0>(params.potential), std::get<1>(params.potential),
           std::get<2>(params.potential)}, params.interaction_start, rng, config},
      rng(rng) {
  normalization = params.U;
 }

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
