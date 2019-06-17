#pragma once
#include "./solver_core.hpp"
#include "./configuration.hpp"
#include "./model.hpp"
#include "./qmc_data.hpp"
#include <list>
#include <functional>

namespace mpi = triqs::mpi;

/** 
 * IMPORTANCE SAMPLING
 * 
 * In importance sampling, we want to sample the integral with n-dimensional 
 * random points $u$ distributed according to a probability density function 
 * $f(u)$, which is as close to the absolute value of the integrand $W_n(u)$ 
 * as possible.
 * The integral is approximated with
 * \[
 *   \sum_{\{u\}} \frac{W_n(u)}{f(u)}.
 * \]
 * 
 * With an appropriate transformation of variables $u --> v$, we can write the 
 * $f(v)$ as a product of independent 1D pdfs:
 * \[
 *   f(v) = \prod_{i=1}^n f_i(v_i).
 * \]
 * Then, we can use the inverse transform sampling to obtain 
 * \[
 *   v_i = F_i^{-1}(l_i),
 * \]
 * where $l$ are uniformly distributed random numbers on $[0,1]$ and $F^{-1}$
 * is the inverse of the cumulative distribution function of $f$.
 * 
 * In short, we have 3 coordinates:
 *   - "l" coordinates defined on [0,1]. We get them from the quasi/pseudo
 *     random number generator.
 *   - "v" coordinates are defined on [t_{interaction start}, 0], which enable
 *     us to separate the N-dimensional model into a product of 1D functions
 *   - "u" coordinates are the original coordinates.
 * $l$, $v$ and $u$ are vectors, their elements are l_i, v_i and u_i, respectively.
 * 
 * This file implements the importance sampling and some wrappers around 
 * `model` (c++2py can't convert object Model for Python interface).
 * File model.cpp implements transformations l --> v --> u and their inverses 
 * as well as $f(l)$, $F^{-1}(l)$ and $F(u)$.
 * Notice, that in the equation above we have $f(u)$ and not $f(l)$. 
 * We provide l-times and convert them to u-times, which are needed for the 
 * integrand, i.e. the `Configuration` weight.
 */


/**
 * Evaluate integrand in point `times_l` in l-domain and reweight it with
 * importance sampling.
 * Returns the integrand and the model weight.
 */
std::vector<dcomplex> solver_core::evaluate_importance_sampling(std::vector<timec_t> times_l, bool do_measure) {
 if (times_l.size() > params.max_perturbation_order)
  TRIQS_RUNTIME_ERROR << "Too many vertices compared to the max perturbation order.";

 // Get u-times
 std::vector<timec_t> times_u = model.l_to_u(times_l);
 
 // Create a vertices list from u-times for the configuration
 auto pot_data = make_potential_data(params.nb_orbitals, params.potential);
 double pot = pot_data.potential_of(0, 0); // We assume 0
 if (pot == 0.)
   TRIQS_RUNTIME_ERROR << "These orbitals have no potential"; // Configuration should never be fed with a zero potential vertex, or it breaks down.
 wrapped_forward_list<vertex_t> vertices_list;
 for (auto it = times_u.begin(); it != times_u.end(); ++it) { 
  vertices_list.insert(0, {0, 0, *it, 0, pot});
 }

 // Calculate
 config.reset_to_vertices(vertices_list);
 config.evaluate();
 config.accept_config();
 
 dcomplex weight = config.accepted_weight;
 if (do_measure) {
  model.evaluate(times_l);
  if (params.method == 1 ) {
   config.accepted_kernels *= std::abs(config.accepted_weight);
   config.accepted_kernels /= std::abs(model.weight);
  }
  if (params.method == 0 ) {
   config.accepted_weight /= std::abs(model.weight);
  }
 }
 measure->accumulate(1);
 return {weight, model.weight};
};

/**
 * Collect kernels for postprocessing.
 * Variable `dummy` is necessary only for the python interface.
 */
void solver_core::collect_sampling_weights(int dummy) {
  mpi::communicator world;
  measure->collect_results(world);
};

/**
 * Set model's parameters
 * We cannot set the parameters when initializing the solver (we provide empty 
 * lists), because we first need the solver in the Python code to obtain them.
 */
void solver_core::set_model(std::vector<std::vector<double>> intervals, std::vector<std::vector<std::vector<double>>> coeff) {
 params.sampling_model_coeff = coeff;
 params.sampling_model_intervals = intervals;
 model = Model(intervals, coeff);
};

/**
 * Returns the model weight in point times_l
 */
dcomplex solver_core::evaluate_model(std::vector<timec_t> times_l) {
 if (times_l.size() > params.max_perturbation_order)
  TRIQS_RUNTIME_ERROR << "Too many vertices compared to the max perturbation order.";

 model.evaluate(times_l);
 return model.weight;
};


void solver_core::init_measure(int dummy) {
  if (params.method == 1) {
   measure = new TwoDetKernelMeasure(&config, &kernels_binning, &pn, &kernels, &kernel_diracs, &nb_kernels);
  }
  if (params.method == 0) {
   measure = new WeightMeasure(&config, &pn, &sn);
  }
}