#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;

// -----------------------
template <typename T>
array<typename det_manip<T>::value_type, 1> cofactor_row(det_manip<T>& matrix, size_t i, size_t n) {
 /* Computes the n-th first cofactors of the i-th row of `matrix`.
  * This does NOT use the inverse matrix.
  */
 array<typename det_manip<T>::value_type, 1> cofactors(n);
 int signs[2] = {1, -1};
 auto x_i = matrix.get_x(i);
 auto y_j = matrix.get_y(0);
 auto y_tmp = y_j;
 matrix.remove(i, 0);
 cofactors(0) = signs[i % 2] * matrix.determinant();
 for (int j = 1; j < n; ++j) {
  y_tmp = matrix.get_y(j-1);
  matrix.change_col(j-1, y_j);
  y_j = y_tmp;
  cofactors(j) = signs[(j + i) % 2] * matrix.determinant();
 }
 matrix.insert(i, n-1, x_i, y_j);
 return cofactors;
};

// -----------------------
class Configuration {

 private:
 bool kernels_comput = true;
 std::pair<double, double> singular_thresholds; // for det_manip
 int max_order = 0;
 array<double, 1> weight_sum;
 array<long, 1> nb_values;
 std::vector<size_t> crea_k_ind;

 // Configuration(const Configuration&) = delete;  // non construction-copyable
 // void operator=(const Configuration&) = delete; // non copyable
 double kernels_evaluate();
 dcomplex keldysh_sum();
 dcomplex keldysh_sum_cofact(int p);

 public:
 std::vector<det_manip<g0_keldysh_alpha_t>>
     matrices; // M matrices for up and down, the first one contains the first annihilation point
 int order;
 array<dcomplex, 2> current_kernels;
 array<dcomplex, 2> accepted_kernels; // kernels of the last accepted config
 dcomplex current_weight;
 dcomplex accepted_weight; // weight of the last accepted config
 array<long, 1> nb_cofact;
 array<long, 1> nb_inverse;

 // registered configurations
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_weight;
 bool stop_register = false;

 // methods
 Configuration(){};
 Configuration(g0_keldysh_alpha_t green_function, std::vector<keldysh_contour_pt> annihila_pts,
               std::vector<keldysh_contour_pt> creation_pts, int max_order,
               std::pair<double, double> singular_thresholds, bool kernels_comput);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void change_left_input(keldysh_contour_pt tau);
 keldysh_contour_pt get_config(int p) const;
 keldysh_contour_pt get_left_input() const;

 void evaluate();
 void accept_config();

 void register_config();

 array<double, 1> get_weight_sum() const { return weight_sum; };
 array<long, 1> get_nb_values() const { return nb_values; };
};

// -----------------------
void nice_print(det_manip<g0_keldysh_alpha_t> det, int p);
