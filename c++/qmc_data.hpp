#pragma once
#include <triqs/det_manip.hpp>
#include <triqs/arrays.hpp>
#include "./model_adaptor.hpp"

// --------------   Point on the Keldysh contour   --------------------------------------

/// A point in time on the double contour, with an additionnal index x_index_t
struct keldysh_contour_pt {
 x_index_t x;  // position on the lattice or orbital index
 qmc_time_t t; // time, in [0, t_max].
 int k_index;  // Keldysh index : 0 (upper contour), or 1 (lower contour)
};

inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<x_index_t, double, int> const &t) {
 return {std::get<0>(t), std::get<1>(t), std::get<2>(t)};
}

/// Comparison (Float is ok in == since we are not doing any operations on them, just store them and compare them in this code).
inline bool operator==(keldysh_contour_pt const &x, keldysh_contour_pt const &y) {
 return (x.t == y.t) and (x.k_index == y.k_index); // and (x.x == y.x); //FIXME
}

/// flip the Keldysh index
inline keldysh_contour_pt flip_index(keldysh_contour_pt const &t) { return {t.x, t.t, 1 - t.k_index}; }

// --------------   G0 adaptor   --------------------------------------

/**
 * Adapt G0_lesser and G0_greater into a function taking two points on the Keldysh contour
 * It is the function that appears in the calculation of the determinant (det_manip, cf below).
 * It is in fact a lambda, but I need its explicit type below to declare det_manip, so I write it explicitly here.
 */
struct g0_keldysh_t {

 g0_adaptor_t g0_lesser;
 g0_adaptor_t g0_greater;
 dcomplex alpha;
 qmc_time_t t_max;

 dcomplex operator()(keldysh_contour_pt const &a, keldysh_contour_pt const &b) const {

  // at equal time, discard Keldysh index and use g_lesser
  // do not put alpha for the time_max even at equal time
  if (a == b) return g0_lesser(a.x, b.x, a.t, b.t) - ((b.t == t_max) ? 0_j : 1_j * alpha);

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

// --------------   data   --------------------------------------

enum spin { up, down };
using triqs::det_manip::det_manip;

/// The data of the QMC
struct qmc_data_t {
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 dcomplex sum_keldysh_indices;                  // Sum of determinants of the last accepted config
 int perturbation_order = 0;                    // the current perturbation order
 const double tmax;                             // time boundary
 const int nb_operators;                        // number of creat/anihil operators in the operator to measure

 qmc_data_t(solve_parameters_t const &params)
    : tmax([&params] {

       std::vector<qmc_time_t> times;
       for (auto spin : {up, down})
        for (auto const &point : params.op_to_measure[spin]) times.push_back(std::get<1>(point));
       return *std::max_element(times.begin(), times.end());

      }()),
      nb_operators([&params] { return params.op_to_measure[up].size() + params.op_to_measure[down].size(); }()) {}
};

// ------------ keldysh sum gray code ------------------------------
dcomplex recompute_sum_keldysh_indices(qmc_data_t *data, const solve_parameters_t *params, int perturbation_order);
