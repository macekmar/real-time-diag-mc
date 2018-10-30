#include "./random_vertex_gen.hpp"

double lorentzian(timec_t t) { return 1 / (t*t + 1); };
double sqrt_lorentzian(timec_t t) { return 1 / std::sqrt(t*t + 1); };

//----------- Uniform sampling --------------
uniform_rvg::uniform_rvg(triqs::mc_tools::random_generator &rng,
            const solve_parameters_t& params)
 : potential_data{params.nb_orbitals,
                  std::get<0>(params.potential),
                  std::get<1>(params.potential),
                  std::get<2>(params.potential)},
   t_max(params.interaction_start), rng(rng)
{};

/// return a random vertex
vertex_t uniform_rvg::operator()() const {
 int k = rng(potential_data.values.size()); // random orbital
 return {potential_data.i_list[k], potential_data.j_list[k], -rng(t_max), 0, potential_data.values[k]};
};

/// return the probability to have chosen vtx in the *current* configuration
double uniform_rvg::probability(const vertex_t& vtx) const {
 return 1 / (potential_data.values.size() * t_max);
};


//----------- Piece-wise preferential sampling --------------
piecewise_rvg::piecewise_rvg(triqs::mc_tools::random_generator &rng,
                const solve_parameters_t& params, const Configuration& config)
 : potential_data{params.nb_orbitals,
                  std::get<0>(params.potential),
                  std::get<1>(params.potential),
                  std::get<2>(params.potential)},
   t_max(params.interaction_start),
   rng(rng),
   config(config),
   gamma(params.ps_gamma),
   f(&sqrt_lorentzian),
   F(&std::asinh)
   //f(&lorentzian),
   //F(&std::atan)
{};

 /// sampling distribution with max = 1
double piecewise_rvg::time_distribution(timec_t t) const {
 auto it = config.times_list().upper_bound(t);

 /// find nearest time
 timec_t t0;
 if (it == config.times_list().begin())
  t0 = *it;
 else if (it == config.times_list().end())
  t0 = *--it;
 else if (std::abs(t - *it) < std::abs(t - *std::prev(it)))
  t0 = *it;
 else
  t0 = *--it;

 return (*f)((t - t0) / gamma);
};

/// return integral (between -t_max and 0) of the sampling distribution
double piecewise_rvg::distrib_norm() const {
 double norm = 0;
 double lower = -t_max;
 double upper = -t_max;
 for (auto it = config.times_list().begin(); it != config.times_list().end(); ++it) {
  lower = upper;
  if (std::next(it) == config.times_list().end())
   upper = 0.0;
  else
   upper = 0.5 * (*it + *std::next(it));
  norm += (*F)((upper - *it) / gamma) - (*F)((lower - *it) / gamma);
 }
 norm *= gamma;
 return norm;
};

/// generates a random time according to the sampling distribution
timec_t piecewise_rvg::random_time_generator() const {
 timec_t time = -rng(t_max);
 double proba = time_distribution(time);
 while (rng(1.0) > proba) {
  time = -rng(t_max);
  proba = time_distribution(time);
 }
 return time;
};

/// return a random vertex
vertex_t piecewise_rvg::operator()() const {
 int k = rng(potential_data.values.size()); // random orbital pair
 return {potential_data.i_list[k], potential_data.j_list[k], random_time_generator(), 0, potential_data.values[k]};
};

/// return the probability to have chosen vtx in the *current* configuration
double piecewise_rvg::probability(const vertex_t& vtx) const {
 return time_distribution(vtx.t) / (distrib_norm() * potential_data.values.size());
};
