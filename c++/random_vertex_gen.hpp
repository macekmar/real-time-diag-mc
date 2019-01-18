#pragma once
#include "./qmc_data.hpp"
#include "./parameters.hpp"
#include "./configuration.hpp"

typedef double (*func_t)(double);

struct RandomVertexGenerator {
 public:
  virtual vertex_t operator()() const = 0;
  virtual double probability(const vertex_t& vtx) const = 0;
  virtual ~RandomVertexGenerator() = default;
};

struct uniform_rvg : RandomVertexGenerator {
 private:
  potential_data_t potential_data;
  const timec_t t_max;
  triqs::mc_tools::random_generator &rng;

 public:
  uniform_rvg(triqs::mc_tools::random_generator &rng, const solve_parameters_t& params);
  vertex_t operator()() const;
  double probability(const vertex_t& vtx) const;
};


template <typename Conf>
struct piecewise_rvg : RandomVertexGenerator {
 private:
  potential_data_t potential_data;
  const timec_t t_max;
  triqs::mc_tools::random_generator &rng;
  const Conf &config;
  const double gamma;
  const func_t f_time;
  const func_t F_time; // should be a primitive of f_time
  const int N; // size of the interacting area
  const func_t f_orb;
  double orb_distrib_norm = 1;

  inline int orbital_distance(orbital_t x1, orbital_t x2) const;
  double time_distribution(timec_t t) const;
  double orbital_distribution(orbital_t x) const;
  double distrib_norm() const;
  timec_t random_time_generator() const;
  int random_coupling_generator() const;

 public:
  piecewise_rvg(triqs::mc_tools::random_generator &rng, const solve_parameters_t& params, const Conf& config);
  vertex_t operator()() const;
  double probability(const vertex_t& vtx) const;
};

