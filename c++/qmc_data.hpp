#pragma once
#include "./model_adaptor.hpp"
#include <triqs/arrays.hpp>
#include <triqs/det_manip.hpp>

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

using triqs::arrays::range;

enum spin { up, down };

struct input_physics_data {

 g0_keldysh_t green_function;
 std::vector<keldysh_contour_pt> tau_list;
 keldysh_contour_pt taup;
 double interaction_start;
 double t_max;
 std::vector<std::size_t> shape_tau_array;
 // int nb_times = 0;
 int rank;
 int min_perturbation_order, max_perturbation_order;
 int op_to_measure_spin; // spin of the operator to measure. Not needed when up/down symmetry. Is used to know which determinant
                         // is the big one.
 array<dcomplex, 1> g0_values;
 const int method;

 // --------
 input_physics_data(const solve_parameters_t *params, g0_t g0_lesser, g0_t g0_greater)
    : method(params->method), interaction_start(params->interaction_start) {

  if (interaction_start < 0) TRIQS_RUNTIME_ERROR << "interaction_time must be positive";

  min_perturbation_order = params->min_perturbation_order;
  max_perturbation_order = params->max_perturbation_order;

  // boundaries
  t_max = *std::max_element(params->measure_times.begin(), params->measure_times.end());
  t_max = std::max(t_max, std::get<1>(params->right_input_points[0]));

  // non interacting Green function
  green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params->alpha, t_max};

  // rank of the Green's function to calculate. For now only 1 is supported.
  if (params->right_input_points.size() % 2 == 0) TRIQS_RUNTIME_ERROR << "There must be an odd number of right input points";
  if (params->right_input_points.size() > 1) TRIQS_RUNTIME_ERROR << "For now only rank 1 Green's functions are supported.";
  rank = params->right_input_points.size() / 2;

  op_to_measure_spin = std::get<0>(params->right_input_points[0]); // for now
  taup = make_keldysh_contour_pt(params->right_input_points[0]);

  // make tau list from left input points lists
  shape_tau_array = {params->measure_times.size(), params->measure_keldysh_indices.size()};

  keldysh_contour_pt tau;
  for (auto time : params->measure_times) {
   for (auto k_index : params->measure_keldysh_indices) {
    tau = {params->measure_state, time, k_index};
    tau_list.emplace_back(tau);
   }
  }

  if (tau_list.size() < 1) TRIQS_RUNTIME_ERROR << "No left input point !";

  // order zero values
  g0_values = array<dcomplex, 1>(tau_list.size());
  for (int i = 0; i < tau_list.size(); ++i) {
   g0_values(i) = green_function(tau_list[i], taup);
  }
 };

 // --------
 array<dcomplex, 1> prefactor() {
  auto output = array<dcomplex, 1>(max_perturbation_order - min_perturbation_order + 1);
  output() = 1.0;

  if (method == 4) output() /= (interaction_start + t_max); // only for cofact formula with additional integral

  dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

  for (int k = 0; k <= max_perturbation_order - min_perturbation_order; ++k) {
   output(k) *= i_n[(k + min_perturbation_order) % 4]; // * i^(k)
  }

  return output;
 };

 // -------
 array<dcomplex, 3> reshape_sn(array<dcomplex, 2> *sn_list) {
  if (second_dim(*sn_list) != tau_list.size()) TRIQS_RUNTIME_ERROR << "The sn list has not the good size to be reshaped.";
  array<dcomplex, 3> sn_array(first_dim(*sn_list), shape_tau_array[0], shape_tau_array[1]);
  int flatten_idx = 0;
  for (int i = 0; i < shape_tau_array[0]; ++i) {
   for (int j = 0; j < shape_tau_array[1]; ++j) {
    sn_array(range(), i, j) = (*sn_list)(range(), flatten_idx);
    flatten_idx++;
   }
  }
  return sn_array;
 };
};

// ------------ keldysh sum gray code ------------------------------
using triqs::det_manip::det_manip;

dcomplex recompute_sum_keldysh_indices(std::vector<det_manip<g0_keldysh_t>> &matrices, int k, int v, int p);
dcomplex recompute_sum_keldysh_indices(std::vector<det_manip<g0_keldysh_t>> &matrices, int k);
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t> &matrix, int k, int p);
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t> &matrix, int k);

void nice_print(det_manip<g0_keldysh_t> det, int p);
