#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;

class Configuration {

 private:
 double singular_threshold = 1e-4; // for det_manip. Not ideal to be defined here
 int op_to_measure_spin = 0;
 double weight_min = 0; // minimal absolute value of the weight

 // Configuration(const Configuration&) = delete;  // non construction-copyable
 // void operator=(const Configuration&) = delete; // non copyable

 public:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 dcomplex weight_value;                         // Sum of determinants of the last accepted config
 int order;

 // registered configurations
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_weight;
 bool stop_register = false;

 Configuration(){};
 Configuration(g0_keldysh_t green_function, const keldysh_contour_pt tau, const keldysh_contour_pt taup,
               const int op_to_measure_spin, double weight_min);
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
 dcomplex weight_evaluate();

 void register_config();

 double get_weight_min() const { return weight_min; };
};
