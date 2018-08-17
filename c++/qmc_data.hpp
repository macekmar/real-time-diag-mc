#pragma once
#include <triqs/gfs.hpp>
#include <triqs/mc_tools.hpp>
#include <triqs/arrays.hpp>
#include <triqs/det_manip.hpp>
#include "./parameters.hpp"

// --------------   Point on the Keldysh contour   ---------------------------

/// A point on the Keldysh contour
struct keldysh_contour_pt {
 orbital_t x;   // orbital index
 spin_t s; // spin (for determinant separation only, the spin is also a component of the orbital index)
 timec_t t;  // time, in [0, t_max].
 int k_index;   // Keldysh index : 0 (upper contour), or 1 (lower contour)
 int rank = -1; // -1 if internal point, else the rank (position) in the correlator to calculate.

 keldysh_contour_pt() = default;
 keldysh_contour_pt(orbital_t x, spin_t s, timec_t t, int k_index, int rank=-1) : x(x), s(s), t(t), k_index(k_index), rank(rank) {};
};

inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<orbital_t, int, timec_t, int> const &t, int r) {
 return {std::get<0>(t), static_cast<spin_t>(std::get<1>(t)), std::get<2>(t), std::get<3>(t), r};
}
inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<orbital_t, spin_t, timec_t, int> const &t, int r) {
 return {std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t), r};
}

/// Comparison (Float is ok in == since we are not doing any operations on them, just store them and compare
/// them in this code).
// Used in tests
inline bool operator==(keldysh_contour_pt const &x, keldysh_contour_pt const &y) {
 return (x.x == y.x) and (x.s == y.s) and (x.t == y.t) and (x.k_index == y.k_index) and (x.rank == y.rank);
}

/// flip the Keldysh index
inline keldysh_contour_pt flip_index(keldysh_contour_pt const &t) { return {t.x, t.s, t.t, 1 - t.k_index, t.rank}; }

// -------------- Vertex ---------------------------------------

/* A vertex is the collection of two (in density-density interaction, four in
 * general) internal contour points with same time and keldysh index, and a
 * potential value. In the case of spin determinant separation, they must be of
 * different spins.
 */
struct vertex_t {
 orbital_t x_up;
 orbital_t x_do;
 timec_t t;
 int k_index;
 double potential;

 keldysh_contour_pt get_up_pt() { return {x_up, up, t, k_index}; };
 keldysh_contour_pt get_down_pt() { return {x_do, down, t, k_index}; };
};

/// Potential
struct potential_data_t {
 int nb_orbitals;
 std::vector<double> values;
 std::vector<orbital_t> i_list;
 std::vector<orbital_t> j_list;
};

/// Vertex random generator
struct vertex_rand_gen {
 potential_data_t potential_data;
 timec_t t_max;
 triqs::mc_tools::random_generator &rng;

 vertex_t operator()() {
  int k = rng(potential_data.values.size());
  return {potential_data.i_list[k], potential_data.j_list[k], -rng(t_max), 0, potential_data.values[k]};
 };

 double size() { return potential_data.values.size() * t_max; };
};

// --------------   G0 adaptor   --------------------------------------


using g0_t = triqs::gfs::gf<triqs::gfs::retime, triqs::gfs::matrix_valued>;

/**
 * Adapt G0_lesser and G0_greater into a function taking two points on the Keldysh contour
 * It is the function that appears in the calculation of the determinant (det_manip, cf below).
 * It is in fact a lambda, but I need its explicit type below to declare det_manip, so I write it explicitly
 * here.
 */
struct g0_keldysh_t {

 g0_t g0_lesser;
 g0_t g0_greater;

 dcomplex operator()(keldysh_contour_pt const &a, keldysh_contour_pt const &b) const {

  // at equal contour time (Float is ok in == since we are not doing any operations on times), discard Keldysh index and use g_lesser
  if (a.t == b.t and a.k_index == b.k_index) return g0_lesser(a.t - b.t)(a.x, b.x);

  //  // mapping: is it lesser or greater?
  //  //  a    b    (a.time > b.time)   L/G ?
  //  //  0    0           1             G
  //  //  0    0           0             L
  //  //  1    1           1             L
  //  //  1    1           0             G
  //  //
  //  //  0    1           *             L
  //  //  1    0           *             G
  bool is_greater = (a.k_index == b.k_index ? (a.k_index xor (a.t > b.t)) : a.k_index);
  return (is_greater ? g0_greater(a.t - b.t)(a.x, b.x) : g0_lesser(a.t - b.t)(a.x, b.x));
 }
};

struct g0_keldysh_alpha_t {

 g0_keldysh_t g0_keldysh;
 dcomplex alpha;
 std::vector<dcomplex> extern_alphas;

 dcomplex operator()(keldysh_contour_pt const &a, keldysh_contour_pt const &b) const {
  if (a.rank < 0) {
   if (a.x == b.x and a.t == b.t and a.rank == b.rank)
    return g0_keldysh(a, b) - 1_j * alpha;
   else
    return g0_keldysh(a, b);
  } else {
   if (a.rank == b.rank)
    return g0_keldysh(a, b) - 1_j * extern_alphas[a.rank];
   else
    return g0_keldysh(a, b);
  }
 }
};

using triqs::det_manip::det_manip;

