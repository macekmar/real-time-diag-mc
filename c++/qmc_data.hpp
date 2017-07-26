#pragma once
#include "./model_adaptor.hpp"
#include <triqs/arrays.hpp>
#include <triqs/det_manip.hpp>

// --------------   Point on the Keldysh contour   --------------------------------------

/// A point in time on the double contour, with an additionnal index x_index_t
struct keldysh_contour_pt {
 x_index_t x;   // position on the lattice or orbital index
 qmc_time_t t;  // time, in [0, t_max].
 int k_index;   // Keldysh index : 0 (upper contour), or 1 (lower contour)
 int rank = -1; // -1 if internal point, else the rank in the correlator to calculate.
};

inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<x_index_t, double, int> const &t) {
 return {std::get<0>(t), std::get<1>(t), std::get<2>(t)};
}

inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<x_index_t, double, int> const &t, int r) {
 return {std::get<0>(t), std::get<1>(t), std::get<2>(t), r};
}

/// Comparison (Float is ok in == since we are not doing any operations on them, just store them and compare
/// them in this code).
inline bool operator==(keldysh_contour_pt const &x, keldysh_contour_pt const &y) {
 return (x.t == y.t) and (x.k_index == y.k_index); // and (x.x == y.x); //FIXME
}

/// flip the Keldysh index
inline keldysh_contour_pt flip_index(keldysh_contour_pt const &t) { return {t.x, t.t, 1 - t.k_index}; }

// --------------   G0 adaptor   --------------------------------------

/**
 * Adapt G0_lesser and G0_greater into a function taking two points on the Keldysh contour
 * It is the function that appears in the calculation of the determinant (det_manip, cf below).
 * It is in fact a lambda, but I need its explicit type below to declare det_manip, so I write it explicitly
 * here.
 */
struct g0_keldysh_t {

 g0_adaptor_t g0_lesser;
 g0_adaptor_t g0_greater;

 dcomplex operator()(keldysh_contour_pt const &a, keldysh_contour_pt const &b) const {

  // at equal time, discard Keldysh index and use g_lesser
  if (a == b) return g0_lesser(a.x, b.x, a.t, b.t);

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
  return (is_greater ? g0_greater(a.x, b.x, a.t, b.t) : g0_lesser(a.x, b.x, a.t, b.t));
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

struct g0_npart {
 /* Implements the function t_1 -> g_0(t_1, ..., t_n | t'_1, ..., t'_n)
  * Where g_0 is the n particles unperturbed Green's function
  * And t_2, ..., t_n are the annihilation contour points, t'_1, ..., t'_n the creation contour points
  */
 // TODO: write tests

 std::vector<keldysh_contour_pt> annihila_pts;
 std::vector<keldysh_contour_pt> creation_pts;
 det_manip<g0_keldysh_t> matrix;

 g0_npart(g0_keldysh_t g0_1part, std::vector<keldysh_contour_pt> annihila_pts,
          std::vector<keldysh_contour_pt> creation_pts)
    : annihila_pts(annihila_pts), creation_pts(creation_pts), matrix{g0_1part, 20} {
  for (size_t i = 1; i < creation_pts.size(); ++i) {
   matrix.insert_at_end(annihila_pts[i], creation_pts[i]);
  }
 }

 dcomplex operator()(keldysh_contour_pt const &tau) {
  matrix.try_insert(0, 0, tau, creation_pts[0]);
  return matrix.determinant();
 }
};

// --------------   data   --------------------------------------

enum spin { up, down };
