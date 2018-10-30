#pragma once
#include "./qmc_data.hpp"
#include "./parameters.hpp"
#include "./configuration.hpp"

typedef double (*time_func_t)(timec_t);

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


struct piecewise_rvg : RandomVertexGenerator {
 private:
  potential_data_t potential_data;
  const timec_t t_max;
  triqs::mc_tools::random_generator &rng;
  const Configuration &config;
  const double gamma;
  const time_func_t f;
  const time_func_t F; // should be a primitive of f

  double time_distribution(timec_t t) const;
  double distrib_norm() const;
  timec_t random_time_generator() const;

 public:
  piecewise_rvg(triqs::mc_tools::random_generator &rng, const solve_parameters_t& params, const Configuration& config);
  vertex_t operator()() const;
  double probability(const vertex_t& vtx) const;
};

