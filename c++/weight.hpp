#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;

class Weight {

 protected:
 double singular_threshold = 1e-4; // for det_manip. Not ideal to be defined here

 public:
 dcomplex value; // Sum of determinants of the last accepted config

 std::vector<std::vector<double>> config_list;
 std::vector<int> config_weight;
 bool stop_register = false;

 virtual void insert(int k, keldysh_contour_pt pt) = 0;
 virtual void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) = 0;
 virtual void remove(int k) = 0;
 virtual void remove2(int k1, int k2) = 0;
 virtual void change_config(int k, keldysh_contour_pt pt) = 0;
 virtual void change_left_input(keldysh_contour_pt tau) = 0;
 virtual void change_right_input(keldysh_contour_pt taup) = 0;
 virtual keldysh_contour_pt get_config(int p) = 0;
 virtual keldysh_contour_pt get_left_input() = 0;
 virtual keldysh_contour_pt get_right_input() = 0;
 virtual dcomplex evaluate() = 0;

 virtual void register_config() = 0;
};

// ------------------------

class two_det_weight : public Weight {

 private:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 const int op_to_measure_spin;

 two_det_weight(const two_det_weight&) = delete; // non construction-copyable
 void operator=(const two_det_weight&) = delete; // non copyable

 public:
 two_det_weight(g0_keldysh_t green_function, const keldysh_contour_pt tau, const keldysh_contour_pt taup, const int op_to_measure_spin);
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

 void register_config();
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
