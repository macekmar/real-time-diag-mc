#pragma once
#include "./parameters.hpp"
#include <triqs/gfs.hpp>
#include <triqs/mc_tools.hpp>

using namespace triqs::gfs;
using gf_latt_time_t = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>;
using gf_latt_time_mat_t = gf<cartesian_product<cyclic_lattice, retime>, matrix_valued, no_tail>;

// --------------  time --------------------------------------

/// Storage of the value of the time.
using qmc_time_t = double;
// ------------------ Single Impurity Matrix -----------------------

#ifdef IMPURITY_MATRIX
using g0_t = gf<retime, matrix_valued>;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t a, x_index_t b, double t, double tp) const { return g0(t - tp)(a, b); }
};

struct random_x_generator {
 int L;
 random_x_generator(g0_t::const_view_type g0, solve_parameters_t const *params) : L(1){};
 // FIXME random_x_generator(g0_t::const_view_type g0, solve_parameters_t const *params) : L(get_target_shape(g0)[0]){};
 x_index_t operator()(triqs::mc_tools::random_generator &rng) const {
  return 0; // point on the lattice
 }
 int size() const { return L; } // size of interacting problem // FIXME modify if only a subset of sites are interacting
};
#endif

// ------------------ Lattice, with scalar GF -----------------------

#ifdef LATTICE_SCALAR
using x_index_t = gf_mesh<cyclic_lattice>::index_t; // The index other than time, can be space coordinate
using g0_t = gf_latt_time_t;
struct g0_adaptor_t {
 g0_t g0;
 auto operator()(x_index_t const &a, x_index_t const &b, double t, double tp) const { return g0(a - b, t - tp); }
};

struct random_x_generator {
 int L;
 random_x_generator(g0_t::const_view_type g0, solve_parameters_t const *params) : L(g0.mesh.size()){};
 x_index_t operator()(triqs::mc_tools::random_generator &rng) const {
  return mindex(rng(L), rng(L), 0); // point on the lattice
 }
 int size() const { return L * L; } // size of interacting problem // FIXME modify if only a subset of sites are interacting
};

#endif

// ------------------ Lattice multiorbital... later ! -----------------------

#ifdef LATTICE_MATRIX
#error "Not implemented"
#endif
