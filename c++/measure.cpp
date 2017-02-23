#include "./measure.hpp"

using triqs::det_manip::det_manip;

// -------- weight_sign_measure -----------

weight_sign_measure::weight_sign_measure(const Weight* weight) : weight(weight) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: weight sign measure" << std::endl;

 value = array<dcomplex, 1>(1); // a single value for a single point measure
 value(0) = 0;
}

void weight_sign_measure::insert(int k, keldysh_contour_pt pt) {}
void weight_sign_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {}
void weight_sign_measure::remove(int k) {}
void weight_sign_measure::remove2(int k1, int k2) {}
void weight_sign_measure::change_config(int k, keldysh_contour_pt pt) {}

void weight_sign_measure::evaluate() { value(0) = weight->value; }

// -------- twodet_cofact_measure -----------

twodet_cofact_measure::twodet_cofact_measure(g0_keldysh_t green_function, const std::vector<keldysh_contour_pt>* tau_list,
                                             const keldysh_contour_pt taup, const int op_to_measure_spin,
                                             const array<dcomplex, 1>* g0_values)
   : green_function(green_function),
     tau_list(tau_list),
     taup(taup),
     op_to_measure_spin(op_to_measure_spin),
     g0_values(g0_values) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: twodet cofact measure" << std::endl;

 value = array<dcomplex, 1>(tau_list->size());
 value() = 0;

 for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);
}

void twodet_cofact_measure::insert(int k, keldysh_contour_pt pt) {
 for (auto& matrix : matrices) matrix.insert(k, k, pt, pt);
}

void twodet_cofact_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto& matrix : matrices) matrix.insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
}

void twodet_cofact_measure::remove(int k) {
 for (auto& matrix : matrices) matrix.remove(k, k);
}

void twodet_cofact_measure::remove2(int k1, int k2) {
 for (auto& matrix : matrices) matrix.remove2(k1, k2, k1, k2);
}

void twodet_cofact_measure::change_config(int k, keldysh_contour_pt pt) {
 for (auto& matrix : matrices) {
  // matrix.change_one_row_and_one_col(k, k, pt, pt);
  matrix.change_row(k, pt);
  matrix.change_col(k, pt);
 }
}

void twodet_cofact_measure::evaluate() {
 keldysh_contour_pt alpha_p_right, alpha_tmp;
 auto matrix_0 = &matrices[op_to_measure_spin];
 auto matrix_1 = &matrices[1 - op_to_measure_spin];
 dcomplex kernel;
 int sign[2] = {1, -1};
 int n = matrices[0].size(); // perturbation order

 if (n == 0) {

  value() = *g0_values;

 } else {

  value() = 0;

  alpha_tmp = taup;
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
    kernel = recompute_sum_keldysh_indices(matrices, n - 1, op_to_measure_spin, p) * sign[(n + p + k_index) % 2];

    for (int i = 0; i < tau_list->size(); ++i) {
     value(i) += green_function((*tau_list)[i], alpha_p_right) * kernel;
    }

    if (k_index == 0) alpha_tmp = alpha_p_right;
   }
  }
  matrix_0->change_col(n - 1, alpha_tmp);
 }
}

// -------- onedet_cofact_measure -----------
/*
onedet_cofact_measure::onedet_cofact_measure(const input_physics_data* physics_params)
   : physics_params(physics_params), green_function(physics_params->green_function), matrix(physics_params->green_function, 100) {

 if (triqs::mpi::communicator().rank() == 0) std::cout << "Measure used: onedet cofact measure" << std::endl;

 value = array<dcomplex, 1>(physics_params->tau_list.size());
 value() = 0;

 matrix.set_singular_threshold(singular_threshold);
}

void onedet_cofact_measure::insert(int k, keldysh_contour_pt pt) { matrix.insert(k, k, pt, pt); }

void onedet_cofact_measure::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 matrix.insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
}

void onedet_cofact_measure::remove(int k) { matrix.remove(k, k); }

void onedet_cofact_measure::remove2(int k1, int k2) { matrix.remove2(k1, k2, k1, k2); }

void onedet_cofact_measure::change_config(int k, keldysh_contour_pt pt) {
 // matrix.change_one_row_and_one_col(k, k, pt, pt);
 matrix.change_row(k, pt);
 matrix.change_col(k, pt);
}

void onedet_cofact_measure::evaluate() {
 keldysh_contour_pt alpha_p_right, alpha_tmp;
 auto matrix_0_v = matrix; // copy
 auto matrix_0 = &matrix_0_v;
 auto matrix_1 = &matrix;
 dcomplex kernel;
 int sign[2] = {1, -1};
 int n = matrix.size(); // perturbation order

 if (n == 0) {

  value() = physics_params->g0_values;

 } else {

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
    kernel = recompute_sum_keldysh_indices(matrix_0, matrix_1, n - 1, p) * sign[(n + p + k_index) % 2];

    for (int i = 0; i < physics_params->tau_list.size(); ++i) {
     value(i) += green_function(physics_params->tau_list[i], alpha_p_right) * kernel;
    }

    if (k_index == 0) alpha_tmp = alpha_p_right;
   }
  }
  matrix_0->change_col(n - 1, alpha_tmp);
 }
}*/
