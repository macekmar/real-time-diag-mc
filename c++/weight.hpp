#pragma once

using triqs::det_manip::det_manip;

class Weight {

 public:
 dcomplex value; // Sum of determinants of the last accepted config

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void change_left_input(keldysh_contour_pt tau);
 void change_right_input(keldysh_contour_pt taup);
 keldysh_contour_pt get_config(int p);
 keldysh_contour_pt get_left_input();
 keldysh_contour_pt get_right_input();
 dcomplex evaluate();
};

// ------------------------

class two_det_weight : public Weight {

 private:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 int op_to_measure_spin;

 two_det_weight(const two_det_weight&) = delete; // non construction-copyable
 void operator=(const two_det_weight&) = delete; // non copyable

 // ------------
 public:
 two_det_weight(const solve_parameters_t* params, const input_physics_data* physics_params) {

  // Initialize the M-matrices. 100 is the initial alocated space.
  for (auto spin : {up, down}) matrices.emplace_back(physics_params->green_function, 100);

  for (auto spin : {up, down}) {
   auto const& ops = params->op_to_measure[spin];
   if (ops.size() == 2) {
    op_to_measure_spin = spin;
    matrices[spin].insert_at_end(make_keldysh_contour_pt(ops[0], 0.5 * (physics_params->t_left_max + physics_params->t_left_min)),
                                 make_keldysh_contour_pt(ops[1], params->weight_time));
   }
  }

  // Initialize value
  value = recompute_sum_keldysh_indices(matrices, 0);
 }

 void insert(int k, keldysh_contour_pt pt) {
  for (auto m : matrices) m.insert(k, k, pt, pt);
 };

 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
  for (auto m : matrices) m.insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 };

 void remove(int k) {
  for (auto m : matrices) m.remove(k, k);
 };

 void remove2(int k1, int k2) {
  for (auto m : matrices) m.remove2(k1, k2, k1, k2);
 };

 void change_config(int k, keldysh_contour_pt pt) {
  for (auto m : matrices) m.change_one_row_and_one_col(k, k, pt, pt);
 };

 void change_left_input(keldysh_contour_pt tau) {
  matrices[op_to_measure_spin].change_row(matrices[op_to_measure_spin].size() - 1, tau);
 };

 void change_right_input(keldysh_contour_pt taup) {
  matrices[1 - op_to_measure_spin].change_col(matrices[1 - op_to_measure_spin].size() - 1, taup);
 };

 keldysh_contour_pt get_config(int p) { return matrices[0].get_x(p); }; // assuming alpha is a single point

 keldysh_contour_pt get_left_input() { return matrices[op_to_measure_spin].get_x(matrices[op_to_measure_spin].size() - 1); };

 keldysh_contour_pt get_right_input() { return matrices[op_to_measure_spin].get_y(matrices[op_to_measure_spin].size() - 1); };

 dcomplex evaluate() {
  return recompute_sum_keldysh_indices(matrices, matrices[1 - op_to_measure_spin].size());
 }
};


// TODO:
// class single_det_weight : Weight {

// det_manip<g0_keldysh_t> matrix; // M matrix
// int op_to_measure_spin;
// double t_left_min, t_left_max; // min and max of the left input time

// qmc_weight_single_det(const qmc_weight_single_det&) = delete; // non construction-copyable
// void operator=(const qmc_weight_single_det&) = delete;        // non copyable

// // ------------
// qmc_weight_single_det(const solve_parameters_t* params, g0_keldysh_t* green_function) : matrix(*green_function, 100) {
//  // Initialize the M-matrix. 100 is the initial alocated space.

//  t_left_max = *std::max_element(params->measure_times.first.begin(), params->measure_times.first.end());
//  t_left_min = *std::min_element(params->measure_times.first.begin(), params->measure_times.first.end());

//  for (auto spin : {up, down}) {
//   auto const& ops = params->op_to_measure[spin];
//   if (ops.size() == 2) {
//    op_to_measure_spin = spin;
//    matrix.insert_at_end(make_keldysh_contour_pt(ops[0], (t_left_min + t_left_max) / 2.),
//                         make_keldysh_contour_pt(ops[1], params->weight_time));
//   }
//  }

//  value = recompute_sum_keldysh_indices(matrix, 0);
// }
//};
