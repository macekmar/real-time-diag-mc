#pragma once
#include "./qmc_data.hpp"
#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>

using triqs::arrays::range;
using triqs::det_manip::det_manip;


// --------------- Measure ----------------

class Measure {

 public:
 virtual array<dcomplex, 1> get_value() = 0;
 virtual void evaluate() = 0;
 virtual void insert(int k, keldysh_contour_pt pt) = 0;
 virtual void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) = 0;
 virtual void remove(int k) = 0;
 virtual void remove2(int k1, int k2) = 0;
 virtual void change_config(int k, keldysh_contour_pt pt) = 0;
};

// ---------------

class twodet_cofact_measure : public Measure {

 private:
 std::vector<det_manip<g0_keldysh_t>>
     matrices; // M matrices for up and down without tau and tau', so they are actually the same...
 g0_keldysh_t green_function;
 input_physics_data* physics_params;
 array<dcomplex, 1> value; // Sum of determinants of the last accepted config

 twodet_cofact_measure(const twodet_cofact_measure&) = delete; // non construction-copyable
 void operator=(const twodet_cofact_measure&) = delete;        // non copyable

 public:
 twodet_cofact_measure(const input_physics_data* physics_params) : green_function(physics_params->green_function) {

  value = array<dcomplex, 1>(physics_params->nb_times);
  value() = 0;

  for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 }

 array<dcomplex, 1> get_value() { return value; }

 void insert(int k, keldysh_contour_pt pt) {
  for (auto spin : {up, down}) {
   matrices[spin].insert(k, k, pt, pt);
  }
 }

 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
  for (auto spin : {up, down}) {
   matrices[spin].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
  }
 }

 void remove(int k) {
  for (auto spin : {up, down}) {
   matrices[spin].remove(k, k);
  }
 }

 void remove2(int k1, int k2) {
  for (auto spin : {up, down}) {
   matrices[spin].remove2(k1, k2, k1, k2);
  }
 }

 void change_config(int k, keldysh_contour_pt pt) {
  for (auto spin : {up, down}) {
   matrices[spin].change_one_row_and_one_col(k, k, pt, pt);
  }
 }

 void evaluate() {
  // Remarks:
  // - why not using the inverse ?
  keldysh_contour_pt alpha_p_right, alpha_tmp;
  auto matrix_0 = &matrices[physics_params->op_to_measure_spin];
  auto matrix_1 = &matrices[1 - physics_params->op_to_measure_spin];
  dcomplex kernel;
  int sign[2] = {1, -1};
  int n = matrices[0].size(); // perturbation order

  if (n == 0) {

   value(range()) = physics_params->order_zero_values;

  } else {

   matrix_0->regenerate();
   matrix_1->regenerate();
   value() = 0;

   alpha_tmp = physics_params->taup;
   matrix_0->roll_matrix(det_manip<g0_keldysh_t>::RollDirection::Left);

   for (int p = 0; p < n; ++p) {
    for (int k_index : {1, 0}) {
     if (k_index == 1) alpha_p_right = matrix_0->get_y((p - 1 + n) % n);
     alpha_p_right = flip_index(alpha_p_right);
     matrix_0->change_one_row_and_one_col(p, (p - 1 + n) % n, flip_index(matrix_0->get_x(p)),
                                          alpha_tmp); // change p keldysh index on the left and p point on the right. p point on
                                                      // the right is effectively changed only when k_index=?.
     matrix_1->change_one_row_and_one_col(p, p, flip_index(matrix_1->get_x(p)), flip_index(matrix_1->get_y(p)));

     // nice_print(*matrix_0, p);
     kernel = recompute_sum_keldysh_indices(matrices, n - 1, physics_params->op_to_measure_spin, p) * sign[(n + p + k_index) % 2];

     for (int i = 0; i < physics_params->nb_times; ++i) {
      value(i) += green_function(physics_params->tau_list[i], alpha_p_right) * kernel;
     }

     if (k_index == 0) alpha_tmp = alpha_p_right;
    }
   }
   matrix_0->change_col(n - 1, alpha_tmp);
  }
 }
};
