#include "./measure.hpp"

using triqs::det_manip::det_manip;

// -------- weight_sign_measure -----------

weight_sign_measure::weight_sign_measure(const input_physics_data* physics_params, const Weight* weight) : weight(weight) {

 if (physics_params->tau_list.size() > 1) TRIQS_RUNTIME_ERROR << "Trying to use a singlepoint measure with multiple input point";
 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: weight sign measure" << std::endl;

 value = array<dcomplex, 1>(physics_params->tau_list.size());
 value() = 0;
}

void weight_sign_measure::insert(int k, keldysh_contour_pt pt) {}
void weight_sign_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {}
void weight_sign_measure::remove(int k) {}
void weight_sign_measure::remove2(int k1, int k2) {}
void weight_sign_measure::change_config(int k, keldysh_contour_pt pt) {}

void weight_sign_measure::evaluate() { value() = weight->value; }

// -------- twodet_single_measure -----------

twodet_single_measure::twodet_single_measure(const input_physics_data* physics_params) : physics_params(physics_params) {

 if (physics_params->tau_list.size() > 1) TRIQS_RUNTIME_ERROR << "Trying to use a singlepoint measure with multiple input point";
 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: twodet single measure" << std::endl;

 value = array<dcomplex, 1>(physics_params->tau_list.size());
 value() = 0;

 for (auto spin : {up, down}) matrices.emplace_back(physics_params->green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);
 matrices[physics_params->op_to_measure_spin].insert_at_end(physics_params->tau_list[0], physics_params->taup);
}

void twodet_single_measure::insert(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  matrices[spin].insert(k, k, pt, pt);
 }
}

void twodet_single_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto spin : {up, down}) {
  matrices[spin].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 }
}

void twodet_single_measure::remove(int k) {
 for (auto spin : {up, down}) {
  matrices[spin].remove(k, k);
 }
}

void twodet_single_measure::remove2(int k1, int k2) {
 for (auto spin : {up, down}) {
  matrices[spin].remove2(k1, k2, k1, k2);
 }
}

void twodet_single_measure::change_config(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  matrices[spin].change_one_row_and_one_col(k, k, pt, pt);
 }
}

void twodet_single_measure::evaluate() {
 value() = recompute_sum_keldysh_indices(matrices, matrices[1 - physics_params->op_to_measure_spin].size());
}

// -------- twodet_multi_measure -----------

twodet_multi_measure::twodet_multi_measure(const input_physics_data* physics_params) : physics_params(physics_params) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: twodet multi measure" << std::endl;

 value = array<dcomplex, 1>(physics_params->tau_list.size());
 value() = 0;

 for (auto spin : {up, down}) matrices.emplace_back(physics_params->green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);
}

void twodet_multi_measure::insert(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  matrices[spin].insert(k, k, pt, pt);
 }
}

void twodet_multi_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto spin : {up, down}) {
  matrices[spin].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 }
}

void twodet_multi_measure::remove(int k) {
 for (auto spin : {up, down}) {
  matrices[spin].remove(k, k);
 }
}

void twodet_multi_measure::remove2(int k1, int k2) {
 for (auto spin : {up, down}) {
  matrices[spin].remove2(k1, k2, k1, k2);
 }
}

void twodet_multi_measure::change_config(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  matrices[spin].change_one_row_and_one_col(k, k, pt, pt);
 }
}

void twodet_multi_measure::evaluate() {
 auto matrix_0 = &matrices[physics_params->op_to_measure_spin];
 auto matrix_1 = &matrices[1 - physics_params->op_to_measure_spin];
 int n = matrices[0].size(); // perturbation order

 if (n == 0) {

  value() = physics_params->g0_values;

 } else {

  matrix_0->regenerate();
  matrix_1->regenerate();
  value() = 0;

  matrix_0->insert(n, n, physics_params->tau_list[0], physics_params->taup);
  value(0) = recompute_sum_keldysh_indices(matrices, n);
  for (int i = 1; i < physics_params->tau_list.size(); ++i) {
   matrix_0->change_row(n, physics_params->tau_list[i]);
   value(i) = recompute_sum_keldysh_indices(matrices, n);
  }
  matrix_0->remove(n, n);
 }
}

// -------- twodet_cofact_measure -----------

twodet_cofact_measure::twodet_cofact_measure(const input_physics_data* physics_params)
   : physics_params(physics_params), green_function(physics_params->green_function) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: twodet cofact measure" << std::endl;

 value = array<dcomplex, 1>(physics_params->tau_list.size());
 value() = 0;

 for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);
}

void twodet_cofact_measure::insert(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  matrices[spin].insert(k, k, pt, pt);
 }
}

void twodet_cofact_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto spin : {up, down}) {
  matrices[spin].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 }
}

void twodet_cofact_measure::remove(int k) {
 for (auto spin : {up, down}) {
  matrices[spin].remove(k, k);
 }
}

void twodet_cofact_measure::remove2(int k1, int k2) {
 for (auto spin : {up, down}) {
  matrices[spin].remove2(k1, k2, k1, k2);
 }
}

void twodet_cofact_measure::change_config(int k, keldysh_contour_pt pt) {
 for (auto spin : {up, down}) {
  // matrices[spin].change_one_row_and_one_col(k, k, pt, pt);
  matrices[spin].change_row(k, pt);
  matrices[spin].change_col(k, pt);
 }
}

void twodet_cofact_measure::evaluate() {
 keldysh_contour_pt alpha_p_right, alpha_tmp;
 auto matrix_0 = &matrices[physics_params->op_to_measure_spin];
 auto matrix_1 = &matrices[1 - physics_params->op_to_measure_spin];
 dcomplex kernel;
 int sign[2] = {1, -1};
 int n = matrices[0].size(); // perturbation order

 if (n == 0) {

  value() = physics_params->g0_values;

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
    // matrix_0->change_one_row_and_one_col(p, (p - 1 + n) % n, flip_index(matrix_0->get_x(p)),
    //                                     alpha_tmp);
    // Change the p keldysh index on the left (row) and the p point on the right (col).
    // The p point on the right is effectively changed only when k_index=1.
    matrix_0->change_row(p, flip_index(matrix_0->get_x(p)));
    matrix_0->change_col((p - 1 + n) % n, alpha_tmp);

    // matrix_1->change_one_row_and_one_col(p, p, flip_index(matrix_1->get_x(p)), flip_index(matrix_1->get_y(p)));
    matrix_1->change_row(p, flip_index(matrix_1->get_x(p)));
    matrix_1->change_col(p, flip_index(matrix_1->get_y(p)));

    // nice_print(*matrix_0, p);
    kernel = recompute_sum_keldysh_indices(matrices, n - 1, physics_params->op_to_measure_spin, p) * sign[(n + p + k_index) % 2];

    for (int i = 0; i < physics_params->tau_list.size(); ++i) {
     value(i) += green_function(physics_params->tau_list[i], alpha_p_right) * kernel;
    }

    if (k_index == 0) alpha_tmp = alpha_p_right;
   }
  }
  matrix_0->change_col(n - 1, alpha_tmp);
 }
}
