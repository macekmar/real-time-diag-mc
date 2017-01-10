#pragma once
#include "./qmc_data.hpp"
#include "./weight.hpp"
#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

using namespace triqs::gfs;
using namespace triqs::statistics;
using triqs::arrays::range;
using triqs::det_manip::det_manip;
namespace mpi = triqs::mpi;


// --------------- Measure ----------------

class qmc_measure {

 private:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 g0_keldysh_t green_function;
 keldysh_contour_pt taup;
 std::vector<keldysh_contour_pt> tau_list;
 int op_to_measure_spin;
 int nb_times;

 array<dcomplex, 1> value;   // Sum of determinants of the last accepted config
 int perturbation_order = 0; // the current perturbation order

 qmc_measure(const qmc_measure&) = delete;    // non construction-copyable
 void operator=(const qmc_measure&) = delete; // non copyable

 public:
 // ----------
 qmc_measure(const solve_parameters_t* params, g0_keldysh_t green_function) : green_function(green_function) {

  nb_times = params->measure_times.first.size();
  value = array<dcomplex, 1>(nb_times);
  value() = 0;

  for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);

  for (auto spin : {up, down}) {
   auto const& ops = params->op_to_measure[spin];
   if (ops.size() == 2) {
    op_to_measure_spin = spin;
    taup = make_keldysh_contour_pt(ops[1], params->measure_times.second);
    for (auto time : params->measure_times.first) {
     tau_list.emplace_back(make_keldysh_contour_pt(ops[0], time));
    }
   }
  }
 }

 array<dcomplex, 1> get_value() { return value; }

 int get_perturbation_order() { return perturbation_order; }

 // ----------
 void insert(int k, keldysh_contour_pt pt) {
  for (auto spin : {up, down}) {
   matrices[spin].insert(k, k, pt, pt);
  }
  perturbation_order++;
 }

 void insert2(int k, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
  for (auto spin : {up, down}) {
   matrices[spin].insert2(k, k + 1, k, k + 1, pt1, pt2, pt1, pt2);
  }
  perturbation_order += 2;
 }

 void remove(int k) {
  for (auto spin : {up, down}) {
   matrices[spin].remove(k, k);
  }
  perturbation_order--;
 }

 void remove2(int k1, int k2) {
  for (auto spin : {up, down}) {
   matrices[spin].remove2(k1, k2, k1, k2);
  }
  perturbation_order -= 2;
 }

 // ----------
 void evaluate() {
  // Remarks:
  // - why not using the inverse ?
  keldysh_contour_pt alpha_p_right, alpha_tmp;
  auto matrix_0 = &matrices[op_to_measure_spin];
  auto matrix_1 = &matrices[1 - op_to_measure_spin];
  dcomplex kernel;
  int sign[2] = {1, -1};
  int n = perturbation_order; // shorter name

  if (n == 0) {

   for (int i = 0; i < nb_times; ++i) {
    value(i) = green_function(tau_list[i], taup);
   }

  } else {

   matrix_0->regenerate();
   matrix_1->regenerate();
   value() = 0;

   alpha_tmp = taup;
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
     kernel = recompute_sum_keldysh_indices(matrices, n - 1, op_to_measure_spin, p) * sign[(n + p + k_index) % 2];

     for (int i = 0; i < nb_times; ++i) {
      value(i) += green_function(tau_list[i], alpha_p_right) * kernel;
     }

     if (k_index == 0) alpha_tmp = alpha_p_right;
    }
   }
   matrix_0->change_col(n - 1, alpha_tmp);
  }
 }
};


// --------------- Accumulator ----------------
// Implements the measure concept for triqs/mc_generic

class qmc_accumulator {

 private:
 qmc_measure* measure;
 qmc_weight* weight;
 array<double, 1>& pn;
 array<dcomplex, 2>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;
 histogram histogram_pn;
 array<dcomplex, 2> sn_node;

 public:
 // ----------
 qmc_accumulator(qmc_measure* measure, qmc_weight* weight, array<double, 1>* pn, array<dcomplex, 2>* sn,
                 array<double, 1>* pn_errors, array<double, 1>* sn_errors, int* nb_measures)
    : measure(measure),
      weight(weight),
      pn(*pn),
      sn(*sn),
      pn_errors(*pn_errors),
      sn_errors(*sn_errors),
      nb_measures(*nb_measures),
      sn_node(*sn) {

  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  sn_node() = 0;
 }

 // ----------
 void accumulate(dcomplex sign) {
  measure->evaluate();

  histogram_pn << measure->get_perturbation_order();
  sn_node(measure->get_perturbation_order(), range()) += measure->get_value() / std::abs(weight->value);
 }

 // ----------
 void collect_results(mpi::communicator c) {

  // We do it with the histogram class
  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  sn = mpi::reduce(sn_node, c);

  // Computing the average and error values
  for (int k = 0; k < size_n; k++) {
   pn(k) = data_histogram_pn(k) / nb_measures; // Average
   sn(k, range()) = sn(k, range()) / data_histogram_pn(k);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   pn_errors(k) =
       (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(k) * (nb_measures - data_histogram_pn(k)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   // sn_errors(k) = (2 / double(pow(data_histogram_pn(k), 2))) *
   //               sqrt((data_histogram_pn(k) - 1) * data_histogram_sn(k) * (data_histogram_pn(k) - data_histogram_sn(k)));
   sn_errors(k) = nan("");
  }
 }
};
