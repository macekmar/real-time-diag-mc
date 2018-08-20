#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

#include <forward_list>
#include <iterator>
#include <utility>

using triqs::det_manip::det_manip;

// -----------------------
template <typename T>
array<typename det_manip<T>::value_type, 1> cofactor_row(det_manip<T>& matrix, size_t i, size_t n) {
 /* Computes the n-th first cofactors of the i-th row of `matrix`.
  * This does NOT use the inverse matrix.
  *
  * In det_manip, vaues of the x-list are constant along rows, while values of
  * the y-list are constant along columns. That is at least the convention I
  * use to name rows and columns.
  */
 array<typename det_manip<T>::value_type, 1> cofactors(n);
 int signs[2] = {1, -1};
 auto x_i = matrix.get_x(i);
 auto y_j = matrix.get_y(0);
 auto y_tmp = y_j;
 matrix.remove(i, 0);
 cofactors(0) = signs[i % 2] * matrix.determinant();
 for (int j = 1; j < n; ++j) {
  y_tmp = matrix.get_y(j - 1);
  matrix.change_col(j - 1, y_j);
  y_j = y_tmp;
  cofactors(j) = signs[(j + i) % 2] * matrix.determinant();
 }
 matrix.insert(i, n - 1, x_i, y_j);
 return cofactors;
};

template <typename T>
array<typename det_manip<T>::value_type, 1> cofactor_col(det_manip<T>& matrix, size_t j, size_t n) {
 /* Computes the n-th first cofactors of the j-th column of `matrix`.
  * This does NOT use the inverse matrix.
  *
  * In det_manip, vaues of the x-list are constant along rows, while values of
  * the y-list are constant along columns. That is at least the convention I
  * use to name rows and columns.
  */
 array<typename det_manip<T>::value_type, 1> cofactors(n);
 int signs[2] = {1, -1};
 auto y_j = matrix.get_y(j);
 auto x_i = matrix.get_x(0);
 auto x_tmp = x_i;
 matrix.remove(0, j);
 cofactors(0) = signs[j % 2] * matrix.determinant();
 for (int i = 1; i < n; ++i) {
  x_tmp = matrix.get_x(i - 1);
  matrix.change_row(i - 1, x_i);
  x_i = x_tmp;
  cofactors(i) = signs[(j + i) % 2] * matrix.determinant();
 }
 matrix.insert(n - 1, j, x_i, y_j);
 return cofactors;
};

// -----------------------
/**
 * Adding some handy methods to std::forward_list.
 *
 * forward_list are singly-chained sequence containers that allow constant time
 * insertion and removal (erase).
 */
template <typename T>
class wrapped_forward_list : public std::forward_list<T> {

 public:

 // insert before index
 void insert(size_t k, T value) {
  auto it = std::forward_list<T>::before_begin();
  std::advance(it, k);
  std::forward_list<T>::insert_after(it, value);
 };

 // erase at index
 void erase(size_t k) {
  auto it = std::forward_list<T>::before_begin();
  std::advance(it, k);
  std::forward_list<T>::erase_after(it);
 };

 // erase at two indices k1 != k2
 void erase(size_t k1, size_t k2) {
  if (k2 < k1) std::swap(k1, k2);
  auto it = std::forward_list<T>::before_begin();
  std::advance(it, k1);
  std::forward_list<T>::erase_after(it);
  std::advance(it, k2 - k1 - 1);
  std::forward_list<T>::erase_after(it);
 };

 // access
 T operator[](size_t k) const {
  auto it = std::forward_list<T>::begin();
  std::advance(it, k);
  return *it;
 };

 // assignement
 T& operator[](size_t k) {
  auto it = std::forward_list<T>::begin();
  std::advance(it, k);
  return *it;
 };
};

// -----------------------
class Configuration {

 // attributes
 private:
 bool kernels_comput = true;
 bool nonfixed_op;
 spin_t spin_dvpt; // tells which matrix is to be developped
 std::pair<double, double> singular_thresholds; // for det_manip
 double cofactor_threshold;
 wrapped_forward_list<double> potential_list;
 double potential = 1.;
 int max_order = 0;
 array<double, 1> weight_sum;
 array<long, 1> nb_values;
 int cycles_trapped = 0;
 int cycles_trapped_thresh = 100;

 public:
 std::vector<det_manip<g0_keldysh_alpha_t>> matrices;
 int order;
 array<dcomplex, 2> current_kernels;
 array<dcomplex, 2> accepted_kernels; // kernels of the last accepted config
 dcomplex current_weight;
 dcomplex accepted_weight; // weight of the last accepted config
 array<long, 1> nb_cofact;
 array<long, 1> nb_inverse;

 // registered configurations
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_mult;
 std::vector<dcomplex> config_weight;

 // methods
 private:
 double kernels_evaluate();
 dcomplex keldysh_sum();
 dcomplex keldysh_sum_cofact(int p); // not used ??

 public:
 Configuration(){};
 Configuration(g0_keldysh_alpha_t green_function, std::vector<keldysh_contour_pt> annihila_pts,
               std::vector<keldysh_contour_pt> creation_pts, int max_order,
               std::pair<double, double> singular_thresholds, bool kernels_comput,
               bool nonfixed_op, int cycles_trapped_thresh);

 //Configuration(const Configuration&) = delete;  // non construction-copyable
 // void operator=(const Configuration&) = delete; // non copyable

 void insert(vertex_t vtx);
 void insert2(vertex_t vtx1, vertex_t vtx2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_vertex(int k, vertex_t vtx);
 vertex_t get_vertex(int p) const;

 void evaluate();
 void accept_config();
 void incr_cycles_trapped();

 array<double, 1> get_weight_sum() const { return weight_sum; };
 array<long, 1> get_nb_values() const { return nb_values; };

 // utility and debug
 std::vector<double> signature();
 void register_accepted_config();
 void register_attempted_config();
 void print();
};

// -----------------------
void nice_print(det_manip<g0_keldysh_alpha_t> det, int p);
