#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;

class Configuration {

 private:
 double singular_threshold = 1e-4; // for det_manip. Not ideal to be defined here
 int max_order = 0;
 array<double, 1> weight_sum;
 array<int, 1> nb_values;

 // Configuration(const Configuration&) = delete;  // non construction-copyable
 // void operator=(const Configuration&) = delete; // non copyable

 public:
 std::vector<det_manip<g0_keldysh_t>>
     matrices;          // M matrices for up and down, the first one contains tau and taup (it is the big one)
 dcomplex weight_value; // Sum of determinants of the last accepted config
 int order;
 array<double, 1> weight_offsets;
 double weight_blur_time;
 array<dcomplex, 2> current_kernels;
 array<dcomplex, 2> accepted_kernels;

 // registered configurations
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_weight;
 bool stop_register = false;

 Configuration(){};
 Configuration(g0_keldysh_t green_function, const keldysh_contour_pt tau, const keldysh_contour_pt taup,
               array<double, 1> weight_offsets, double weight_blur_time, int max_order);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void change_left_input(keldysh_contour_pt tau);
 void change_right_input(keldysh_contour_pt taup);
 keldysh_contour_pt get_config(int p) const;
 keldysh_contour_pt get_left_input() const;
 keldysh_contour_pt get_right_input() const;

 dcomplex keldysh_sum();
 dcomplex keldysh_sum_cofact(int p);
 array<dcomplex, 2> kernels_evaluate_cofact();
 array<dcomplex, 2> kernels_evaluate_inverse();
 double weight_kernels();
 dcomplex weight_evaluate();

 void register_config();

 array<double, 1> get_weight_sum() const { return weight_sum; };
 array<int, 1> get_nb_values() const { return nb_values; };
};

void nice_print(det_manip<g0_keldysh_t> det, int p);
