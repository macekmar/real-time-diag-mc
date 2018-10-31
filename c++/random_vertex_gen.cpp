#include "./random_vertex_gen.hpp"

double lorentzian(double x) { return 1.0 / (x*x + 1.0); };
double sqrt_lorentzian(double x) { return 1.0 / std::sqrt(x*x + 1.0); };

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
   f_time(&sqrt_lorentzian),
   F_time(&std::asinh),
   f_orb(&sqrt_lorentzian),
   //f_time(&lorentzian),
   //F_time(&std::atan)
   N(std::sqrt(potential_data.values.size()))
{std::cout << N << std::endl;};

inline int piecewise_rvg::orbital_distance(orbital_t x1, orbital_t x2) const {
 int x1_x = x1 % N;
 int x1_y = x1 / N;
 int x2_x = x2 % N;
 int x2_y = x2 / N;
 return std::max(std::abs(x1_x - x2_x), std::abs(x1_y - x2_y));
};

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

 return (*f_time)((t - t0) / gamma);
};

double piecewise_rvg::orbital_distribution(orbital_t x) const {
 int dist = 2*N;
 for (auto it = config.orbitals_list().begin(); it != config.orbitals_list().end(); ++it) {
  dist = std::min(dist, orbital_distance(x, *it));
 }
 return (*f_orb)(static_cast<double>(dist) / gamma);
};

/// return integral (between -t_max and 0) of the sampling distribution
double piecewise_rvg::distrib_norm() const {

 // time integral
 double norm_time = 0;
 double lower = -t_max;
 double upper = -t_max;
 for (auto it = config.times_list().begin(); it != config.times_list().end(); ++it) {
  lower = upper;
  if (std::next(it) == config.times_list().end())
   upper = 0.0;
  else
   upper = 0.5 * (*it + *std::next(it));
  norm_time += (*F_time)((upper - *it) / gamma) - (*F_time)((lower - *it) / gamma);
 }
 norm_time *= gamma;

 // orbital sum
 double norm_orb = 0;
 for (orbital_t x = 0; x != potential_data.values.size(); ++x) {
  norm_orb += orbital_distribution(x);
 }

 return norm_time * norm_orb;
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

int piecewise_rvg::random_coupling_generator() const {
 int k = rng(potential_data.values.size());
 double proba = orbital_distribution(potential_data.i_list[k]);
 while (rng(1.0) > proba) {
  k = rng(potential_data.values.size());
  proba = orbital_distribution(potential_data.i_list[k]);
 }
 return k;
};

/// return a random vertex
vertex_t piecewise_rvg::operator()() const {
 int k = random_coupling_generator();
 return {potential_data.i_list[k], potential_data.j_list[k], random_time_generator(), 0, potential_data.values[k]};
};

/// return the probability to have chosen vtx in the *current* configuration
double piecewise_rvg::probability(const vertex_t& vtx) const {
 return time_distribution(vtx.t) * orbital_distribution(vtx.x_up) / distrib_norm();
};
