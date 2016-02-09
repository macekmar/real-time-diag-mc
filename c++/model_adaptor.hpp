#pragma once
#include <triqs/gfs.hpp>
#include "./parameters.hpp"

using namespace triqs::gfs;
using gf_latt_time_t = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>;
using gf_latt_time_mat_t = gf<cartesian_product<cyclic_lattice, retime>, matrix_valued, no_tail>;

// --------------  time --------------------------------------

/// Storage of the value of the time. A double
using qmc_time_t = double;

#define IMPURITY_SCALAR

// ------------------ Single Impurity Scalar -----------------------

#ifdef IMPURITY_SCALAR
using x_index_t = int; // unused
using g0_t = gf<retime, scalar_valued, no_tail>;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t, x_index_t, double t, double tp) const { return g0(t - tp); }

 x_index_t get_random_x(triqs::mc_tools::random_generator &rng, qmc_data_t const * data) const { return 0; }
 int get_random_n_values() const { return 1;}

};

x_index_t get_random_x(triqs::mc_tools::random_generator &rng, const solve_parameters_t *params) { return rng(params->L); }
#endif

// ------------------ Single Impurity Matrix -----------------------

#ifdef IMPURITY_MATRIX
using x_index_t = int;
using g0_t = gf<retime, matrix_valued, no_tail>;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t a, x_index_t b, double t, double tp) const { return g0(t - tp)(a, b); }
};

x_index_t get_random_x(triqs::mc_tools::random_generator &rng, const solve_parameters_t *params) { return rng(params->L); }
#endif

// ------------------ Lattice, with scalar GF -----------------------

#ifdef LATTICE_SCALAR
using x_index_t = gf_mesh<cyclic_lattice>::index_t; // The index other than time, can be space coordinate
using g0_t = gf_latt_time_t;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t const &a, x_index_t const &b, double t, double tp) const { return g0(a - b, t - tp); }

 x_index_t get_random_x(triqs::mc_tools::random_generator &rng, qmc_data_t const *data) {
  auto L = data->g0_lesser_adaptor.g0_lesser.mesh();
  return rng(L);
 }
};

#endif

// ------------------ Lattice multiorbital... later ! -----------------------

#ifdef LATTICE_MATRIX
#error "Not implemented"
#endif
