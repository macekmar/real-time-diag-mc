#pragma once
#include <triqs/gfs.hpp>
#include "./parameters.hpp"

using namespace triqs::gfs;
using gf_latt_time_t = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>;
using gf_latt_time_mat_t = gf<cartesian_product<cyclic_lattice, retime>, matrix_valued, no_tail>;

// --------------  time --------------------------------------

/// Storage of the value of the time. A double
using qmc_time_t = double;

// ----------------------------------------------------

// three cases
#define IMPURITY_SCALAR
#ifdef IMPURITY_SCALAR
using x_index_t = int; // unused
using g0_t = gf<retime, scalar_valued, no_tail>;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t, x_index_t, double t, double tp) const { return g0(t - tp); }
};

x_index_t get_random_x(triqs::mc_tools::random_generator &rng, const solve_parameters_t *params) { return rng(params->L); }
#endif

#ifdef IMPURITY_MATRIX
using x_index_t = int;
using g0_t = gf<retime, matrix_valued, no_tail>;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t a, x_index_t b, double t, double tp) const { return g0(t - tp)(a, b); }
};

x_index_t get_random_x(triqs::mc_tools::random_generator &rng, const solve_parameters_t *params) { return rng(params->L); }
#endif

#ifdef LATTICE_SCALAR
using x_index_t = gf_mesh<cyclic_lattice>::index_t; // The index other than time, can be space coordinate
using g0_t = gf_latt_time_t;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t const &a, x_index_t const &b, double t, double tp) const { return g0(a - b, t - tp); }
};

x_index_t get_random_x(triqs::mc_tools::random_generator &rng) { auto L = g0.; return rng(L); }
#endif

#ifdef LATTICE_MATRIX
#error "Not implemented"
#endif


#ifdef LATTICE 
#endif

