#pragma once
#include <triqs/gfs.hpp>
#include "./parameters.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;
using namespace triqs::gfs;

using gf_latt_time_t = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>;

// --------------  time --------------------------------------

/// Storage of the value of the time. A double, without any possible operations and cast.
class qmc_time_t {
 double t = 0;

 public:
 qmc_time_t() = default;
 explicit qmc_time_t(double x) : t(x) {}
 explicit operator double() const { return t; }
 friend bool operator==(qmc_time_t x, qmc_time_t y) { return x.t == y.t; }
 friend bool operator>(qmc_time_t x, qmc_time_t y) { return x.t > y.t; }
};

// ----------------------------------------------------

#define LATTICE

// three cases
#ifdef LATTICE
using gf0_t =gf_latt_time_t; // the type of g0< and g0>
using space_index_t = gf_mesh<cyclic_lattice>::index_t; // The index other than time, can be space coordinate
template <typename G, typename X, typename T> auto _call(G &g, X &&x, T &&t) { return g(x, t); }
template <typename G, typename X> auto _call0(G &g, X &&x) { return g(mindex(0, 0, 0), 0.0); }
#endif

#ifdef IMPURITY_SCALAR
using gf0_t = gf<retime, scalar_valued, no_tail>;
using space_index_t = int;
template <typename G, typename X, typename T> auto _call(G &g, X &&x, T &&t) { return g(t); }
template <typename G, typename X> auto _call0(G &g, X &&x) { return g(0.0); }
#endif

#ifdef IMPURITY_MATRIX
using gf0_t = gf<retime, matrix_valued, no_tail>;
using space_index_t = int;
template <typename G, typename X, typename T> auto _call(G &g, X &&x, T &&t) { return g(t)(x[0],x[1]); }
template <typename G, typename X> auto _call0(G &g, X &&x) { return g(0.0)(x[0],x[1]); }
#endif


/// A point in time on the double contour.
struct point {
 space_index_t x; // position on the lattice.
 qmc_time_t time; // time, in [0, t_max].
 int k_index;     // Keldysh index : 0 (upper contour), or 1 (lower contour)
};
inline bool operator==(point const &x, point const &y) { return (x.time == y.time) and (x.k_index == y.k_index) and (x.x == y.x); }

/// flip the Keldysh index
inline point flip_index(point const &t) { return {t.x, t.time, 1 - t.k_index}; }

// --------------   G0 adaptor   --------------------------------------

// G0 in Keldysh space from G0_lesser and G0_greater.
// It is the function that appears in the calculation of the determinant
// It is a (kind of) lambda, but I need its explicit type below to declare det_manip, so I write it explicitely here.
struct g0_keldysh_t {

 gf0_t const &g0_lesser;
 gf0_t const &g0_greater;
 dcomplex alpha;
 qmc_time_t t_max;

 dcomplex operator()(point const &x, point const &y) const {

  // do not put alpha for the time_max even at equal time
  if (x == y) return _call0(g0_lesser,x) - ((y.time == t_max) ? 0_j : 1_j * alpha);
  auto dx = x.x - y.x;

  double dt = double(x.time) - double(y.time);
  // mapping: is it lesser or greater ?
  //  x    y    (x.time > y.time)   L/G ?
  //  0    0           0             L
  //  0    0           1             G
  //  1    1           0             G
  //  1    1           1             L
  //
  //  0    1           *            L
  //  1    0           *            G
  bool is_lesser = (x.k_index == y.k_index ? (x.k_index xor (x.time > y.time)) : y.k_index);
  return (is_lesser ? _call(g0_lesser, dx, dt) : _call(g0_greater, dx, dt));

  // old code. slower because more tests 
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
 dcomplex sum_keldysh_indices = 1;
 int perturbation_order = 0;
};


