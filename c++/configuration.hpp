#pragma once
#include "./qmc_data.hpp"
#include <triqs/det_manip.hpp>

#include <forward_list>
#include <set>
#include <iterator>
#include <utility>
#include <list>

using triqs::det_manip::det_manip;

// -----------------------
 /**
  * Computes the n-th first cofactors of the i-th row of `matrix`.
  * This does NOT use the inverse matrix.
  *
  * In det_manip, vaues of the x-list are constant along rows, while values of
  * the y-list are constant along columns. That is at least the convention I
  * use to name rows and columns.
  */
template <typename T>
array<typename det_manip<T>::value_type, 1> cofactor_row(det_manip<T>& matrix, size_t i, size_t n) {
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

 /**
  * Computes the n-th first cofactors of the j-th column of `matrix`.
  * This does NOT use the inverse matrix.
  *
  * In det_manip, vaues of the x-list are constant along rows, while values of
  * the y-list are constant along columns. That is at least the convention I
  * use to name rows and columns.
  */
template <typename T>
array<typename det_manip<T>::value_type, 1> cofactor_col(det_manip<T>& matrix, size_t j, size_t n) {
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

 // insert two values so that they will be at positions `k1` and `k2`.
 void insert2(size_t k1, size_t k2, T value1, T value2) {
  if (k2 < k1) {
   std::swap(k1, k2);
   std::swap(value1, value2);
  }
  auto it = std::forward_list<T>::before_begin();
  std::advance(it, k1);
  std::forward_list<T>::insert_after(it, value1);
  std::advance(it, k2 - k1);
  std::forward_list<T>::insert_after(it, value2);
 };

 // erase at index
 void erase(size_t k) {
  auto it = std::forward_list<T>::before_begin();
  std::advance(it, k);
  std::forward_list<T>::erase_after(it);
 };

 // erase at two indices k1 != k2
 void erase2(size_t k1, size_t k2) {
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
 protected:
 solve_parameters_t params; // needs to be a non-const value, because of this two-step construction of SolverCore.
 spin_t spin_dvpt; // tells which matrix is to be developped
 double cofactor_threshold;
 wrapped_forward_list<double> potential_list;
 std::set<timec_t> times_list_; // sorted container
 wrapped_forward_list<orbital_t> orbitals_list_;
 std::list<keldysh_contour_pt> creation_pts, annihila_pts;
 double potential = 1.;
 int cycles_trapped = 0;
 
 public:
 std::vector<det_manip<g0_keldysh_alpha_t>> matrices;
 int order;
 array<dcomplex, 2> current_kernels;
 array<dcomplex, 2> accepted_kernels; // kernels of the last accepted config
 dcomplex current_weight; // weights have to be complex as they are used for sn measure in the oldway method
 dcomplex accepted_weight; // weight of the last accepted config
 array<long, 1> nb_cofact;
 array<long, 1> nb_inverse;

 // registered configurations
 std::vector<std::vector<double>> config_list;
 std::vector<int> config_mult;
 std::vector<dcomplex> config_weight;

 // methods
 protected:
 double kernels_evaluate();
 dcomplex keldysh_sum();
 dcomplex keldysh_sum_cofact(int p); // not used ??

 public:
 Configuration(){}; // needed for the two-step construction of SolverCore
 Configuration(g0_keldysh_alpha_t green_function, const solve_parameters_t &params);

 /*
 Configuration(const Configuration&) = delete;  // non construction-copyable
 void operator=(const Configuration&) = delete; // non copyable
 Configuration(Configuration&&) = default; // construction-movable
 Configuration& operator=(Configuration&&) = default; // movable
 */
 // For some weird reason, `Configuration` needs to be movable to be
 // non-copyable. If not, compilation fails saying that we tried to copy
 // `det_manip` which is non-copyable ??!!
 // Marjan: Compiles and passes tests without these lines just fine!

 void insert(int k, vertex_t vtx);
 void insert2(int k1, int k2, vertex_t vtx1, vertex_t vtx2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_vertex(int k, vertex_t vtx);
 vertex_t get_vertex(int p) const;
 inline timec_t get_time(int k) const;

 void remove_all() { while (order > 0) remove(0); };
 void insert_vertices(std::list<vertex_t> vertices);
 void reset_to_vertices(std::list<vertex_t> vertices);

 virtual void evaluate() = 0;
 void accept_config();
 void incr_cycles_trapped();

 // getters
 const std::set<timec_t>& times_list() const {return times_list_;};
 const wrapped_forward_list<orbital_t>& orbitals_list() const {return orbitals_list_;};
 std::list<vertex_t> vertices_list();

 // utility and debug
 std::vector<double> signature();
 void register_accepted_config();
 void register_attempted_config();
 void set_ops();
 void set_default_values();
 void print();
};

// -----------------------
void nice_print(det_manip<g0_keldysh_alpha_t> det, int p);

// ------------ QMC ------------ ----------------------------------------------
class ConfigurationQMC : public Configuration {
  public:
  ConfigurationQMC(){};
  ConfigurationQMC(g0_keldysh_alpha_t green_function, const solve_parameters_t &params) : Configuration(green_function, params) {evaluate(); accept_config();};

  void evaluate();
};

// ------------ Auxillary MC --------------------------------------------------
class ConfigurationAuxMC : public Configuration {
  public:
  ConfigurationQMC config_qmc;

  ConfigurationAuxMC(){};
  ConfigurationAuxMC(g0_keldysh_alpha_t green_function, const solve_parameters_t &params) : Configuration(green_function, params), config_qmc(green_function, params) {evaluate(); accept_config();};

  void evaluate();
  dcomplex _eval(std::vector<std::tuple<orbital_t, orbital_t, timec_t>> vertices);
};
