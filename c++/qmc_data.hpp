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

inline keldysh_contour_pt make_keldysh_contour_pt(std::tuple<x_index_t, int> const &t, double const time) {
 return {std::get<0>(t), time, std::get<1>(t)};
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
 double t_max;
 double t_left_min, t_left_max;
 int nb_times = 0;
 int nb_operators;
 int min_perturbation_order, max_perturbation_order;
 spin op_to_measure_spin; // spin of the operator to measure. Not needed when up/down symmetry. Is used to know which determinant
                          // is the big one.
 array<dcomplex, 1> order_zero_values;
 const int method;

 input_physics_data(const solve_parameters_t *params, g0_t g0_lesser, g0_t g0_greater) : method(params->method) {

  min_perturbation_order = params->min_perturbation_order;
  max_perturbation_order = params->max_perturbation_order;

  // boundaries
  t_max = *std::max_element(params->measure_times.first.begin(), params->measure_times.first.end());
  t_max = std::max(t_max, params->measure_times.second);
  t_left_max = *std::max_element(params->measure_times.first.begin(), params->measure_times.first.end());
  t_left_min = *std::min_element(params->measure_times.first.begin(), params->measure_times.first.end());

  // non interacting Green function
  green_function = g0_keldysh_t{g0_adaptor_t{g0_lesser}, g0_adaptor_t{g0_greater}, params->alpha, t_max};

  // number of operators in the correlator to measure. For now only 2 is supported
  nb_operators = params->op_to_measure[up].size() + params->op_to_measure[down].size();

  // input times
  for (auto spin : {up, down}) {
   auto const &ops = params->op_to_measure[spin];
   if (ops.size() == 2) {
    op_to_measure_spin = spin;
    taup = make_keldysh_contour_pt(ops[1], params->measure_times.second);
    for (auto time : params->measure_times.first) {
     tau_list.emplace_back(make_keldysh_contour_pt(ops[0], time));
     nb_times++;
    }
   }
  }

  if (nb_times < 1) TRIQS_RUNTIME_ERROR << "No left input times !";

  // order zero values
  order_zero_values = array<dcomplex, 1>(nb_times);
  for (int i = 0; i < nb_times; ++i) {
   order_zero_values(i) = green_function(tau_list[i], taup);
  }
 };

 double left_time_normalize(double left_time) const {
  return (left_time - t_left_min) / (t_left_max - t_left_min); // no divide by zero check !
 };

 double left_time_denormalize(double norm_time) const { return norm_time * (t_left_max - t_left_min) + t_left_min; };

 array<dcomplex, 1> prefactor() {
  auto output = array<dcomplex, 1>(max_perturbation_order - min_perturbation_order + 1);
  output() = 1.0;

  if (method == 4) output() /= (t_left_max - t_left_min); // only for cofact formula with additional integral

  dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

  for (int k = 0; k <= max_perturbation_order - min_perturbation_order; ++k) {
   output(k) *= i_n[(k + min_perturbation_order) % 4]; // * i^(k)
  }

  if (nb_operators == 2)
   output() *= i_n[3]; // additional factor of -i
  else if (nb_operators == 4)
   output() *= i_n[2]; // additional factor of -1=i^6
  else
   TRIQS_RUNTIME_ERROR << "Operator to measure not recognised.";

  return output;
 };
};

// ------------ keldysh sum gray code ------------------------------
using triqs::det_manip::det_manip;

dcomplex recompute_sum_keldysh_indices(std::vector<det_manip<g0_keldysh_t>> &matrices, int k, int v, int p);
dcomplex recompute_sum_keldysh_indices(std::vector<det_manip<g0_keldysh_t>> &matrices, int k);
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t> &matrix, int k, int p);
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t> &matrix, int k);

void nice_print(det_manip<g0_keldysh_t> det, int p);
