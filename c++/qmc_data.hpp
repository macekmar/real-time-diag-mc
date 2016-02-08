#pragma once
#include <triqs/det_manip.hpp>
#include "./model_adaptor.hpp"

using triqs::det_manip::det_manip;

/// A double_contour_pt in time on the double contour.
struct keldysh_contour_pt {
 x_index_t x;  // position on the lattice.
 qmc_time_t t; // time, in [0, t_max].
 int k_index;  // Keldysh index : 0 (upper contour), or 1 (lower contour)
};
inline bool operator==(keldysh_contour_pt const &x, keldysh_contour_pt const &y) { return (x.t == y.t) and (x.k_index == y.k_index) and (x.x == y.x); }

/// flip the Keldysh index
inline keldysh_contour_pt flip_index(keldysh_contour_pt const &t) { return {t.x, t.t, 1 - t.k_index}; }

// --------------   G0 adaptor   --------------------------------------

// G0 in Keldysh space from G0_lesser and G0_greater.
// It is the function that appears in the calculation of the determinant
// It is a (kind of) lambda, but I need its explicit type below to declare det_manip, so I write it explicitely here.
struct g0_keldysh_t {

 g0_adaptor_t const &g0_lesser;
 g0_adaptor_t const &g0_greater;
 dcomplex alpha;
 qmc_time_t t_max;

 dcomplex operator()(keldysh_contour_pt const &a, keldysh_contour_pt const &b) const {

  // do not put alpha for the time_max even at equal time
  if (a == b) return g0_lesser(a.x, b.x, a.t, b.t) - ((b.t == t_max) ? 0_j : 1_j * alpha);

  // mapping: is it lesser or greater ?
  //  x    y    (x.time > y.time)   L/G ?
  //  0    0           0             L
  //  0    0           1             G
  //  1    1           0             G
  //  1    1           1             L
  //
  //  0    1           *            L
  //  1    0           *            G
  bool is_lesser = (a.k_index == b.k_index ? (a.k_index xor (a.t> b.t)) : b.k_index);
  return (is_lesser ? g0_lesser(a.x, b.x, a.t, b.t) : g0_greater(a.x, b.x, a.t, b.t));

  // old code. slower because more tests FIXME: REMOVE AFTER DEBUG 
  // if (x.k_index == 0 && y.k_index == 0) return (x.time > y.time ? g0_greater(dr, dt) : g0_lesser(dr, dt));
  // if (x.k_index == 1 && y.k_index == 1) return (x.time > y.time ? g0_lesser(dr, dt) : g0_greater(dr, dt));
  // if (x.k_index == 1 && y.k_index == 0) return g0_greater(dr, dt);
  // return g0_lesser(dr, dt);
 }
 };

// --------------   data   --------------------------------------

enum spin {up, down};

struct qmc_data_t { 
 std::vector<det_manip<g0_keldysh_t>> matrices; ///< M-matrices for up and down
 dcomplex sum_keldysh_indices = 0; // g0 ? FIXME: call for once
 int perturbation_order = 0;
};


