#pragma once
#include "./configuration.hpp"
#include <triqs/arrays.hpp>
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

// using namespace triqs::gfs;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;

struct KernelBinning {
 std::vector<array<dcomplex, 3>> values; // 3D: binning, p, keldysh index
 double t_min, t_max, bin_length;

 KernelBinning(){};
 KernelBinning(double t_min, double t_max, int nb_bins, int max_order)
    : t_max(t_max), t_min(t_min), bin_length((t_max - t_min) / nb_bins) {
  for (int k = 1; k <= max_order; ++k) {
   values.emplace_back(array<dcomplex, 3>(nb_bins, k, 2));
   values[k - 1]() = 0;
  }
 };

 void add(keldysh_contour_pt alpha, int order, int p, dcomplex value) {
  assert(t_min <= alpha.t);
  assert(alpha.t <= t_max);
  int bin = int((alpha.t - t_min) / bin_length);
  values[order - 1](bin, p, alpha.k_index) += value;
 };
};


class WeightSignMeasure {

 private:
 Configuration& config;
 array<int, 1>& pn;
 array<dcomplex, 2>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;
 histogram histogram_pn;
 array<dcomplex, 2> sn_node;

 public:
 // ----------
 WeightSignMeasure(Configuration* config, array<int, 1>* pn, array<dcomplex, 2>* sn,
                   array<double, 1>* pn_errors, array<double, 1>* sn_errors, int* nb_measures)
    : config(*config),
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

  histogram_pn << config.order;
  // FIXME: first dim of sn_node is out of range if min_perturbation_order is not zero !
  sn_node(config.order, range()) += config.weight_value / std::abs(config.weight_value);
 }

 // ----------
 void collect_results(mpi::communicator c) {

  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  sn = mpi::reduce(sn_node, c);

  // Computing the average and error values
  for (int k = 0; k < size_n; k++) {
   pn(k) = data_histogram_pn(k);
   if (pn(k) == 0) pn(k) = 1; // Avoids divide by zero, sn(k) should be zero in this case
   sn(k, range()) = sn(k, range()) / pn(k);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   // pn_errors(k) =
   //    (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(k) * (nb_measures -
   //    data_histogram_pn(k)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   // sn_errors(k) = (2 / double(pow(data_histogram_pn(k), 2))) *
   //               sqrt((data_histogram_pn(k) - 1) * data_histogram_sn(k) * (data_histogram_pn(k) -
   //               data_histogram_sn(k)));
   sn_errors(k) = nan("");
   pn_errors(k) = nan("");
  }
 }
};

class TwoDetCofactMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels;
 array<int, 1>& pn;
 array<dcomplex, 2>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;
 histogram histogram_pn;
 array<dcomplex, 2> sn_node;
 const std::vector<keldysh_contour_pt>* tau_list;
 const keldysh_contour_pt taup;
 const int op_to_measure_spin;
 const array<dcomplex, 1>* g0_values;
 g0_keldysh_t green_function;

 public:
 // ----------
 TwoDetCofactMeasure(Configuration* config, KernelBinning* kernels, array<int, 1>* pn, array<dcomplex, 2>* sn,
                     array<double, 1>* pn_errors, array<double, 1>* sn_errors, int* nb_measures,
                     const std::vector<keldysh_contour_pt>* tau_list, const keldysh_contour_pt taup,
                     const int op_to_measure_spin, const array<dcomplex, 1>* g0_values,
                     g0_keldysh_t green_function)
    : config(*config),
      kernels(*kernels),
      pn(*pn),
      sn(*sn),
      pn_errors(*pn_errors),
      sn_errors(*sn_errors),
      nb_measures(*nb_measures),
      sn_node(*sn),
      tau_list(tau_list),
      taup(taup),
      op_to_measure_spin(op_to_measure_spin),
      g0_values(g0_values),
      green_function(green_function) {

  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  sn_node() = 0;
 }

 // ----------
 void accumulate(dcomplex sign) {

  histogram_pn << config.order;

  keldysh_contour_pt alpha_p_right, alpha_tmp;
  auto matrix_0 = &(config.matrices[op_to_measure_spin]);
  auto matrix_1 = &(config.matrices[1 - op_to_measure_spin]);
  dcomplex kernel;
  int signs[2] = {1, -1};
  int n = config.order; // perturbation order
  array<dcomplex, 1> value(tau_list->size());

  if (n == 0) {

   value() = *g0_values;

  } else {

   value() = 0;
   auto tau_weight = config.get_left_input();
   matrix_0->remove(n, n); // remove tau_w and taup

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

     // matrix_1->change_one_row_and_one_col(p, p, flip_index(matrix_1->get_x(p)),
     // flip_index(matrix_1->get_y(p)));
     matrix_1->change_row(p, flip_index(matrix_1->get_x(p)));
     matrix_1->change_col(p, flip_index(matrix_1->get_y(p)));

     // nice_print(*matrix_0, p);
     kernel = recompute_sum_keldysh_indices(config.matrices, n - 1, op_to_measure_spin, p) *
              signs[(n + p + k_index) % 2];
     kernels.add(alpha_p_right, n, p, kernel / std::abs(config.weight_value));

     for (int i = 0; i < tau_list->size(); ++i) {
      value(i) += green_function((*tau_list)[i], alpha_p_right) * kernel;
     }

     if (k_index == 0) alpha_tmp = alpha_p_right;
    }
   }
   matrix_0->change_col(n - 1, alpha_tmp);

   matrix_0->insert(n, n, tau_weight, taup);
  }


  // FIXME: first dim of sn_node is out of range if min_perturbation_order is not zero !
  sn_node(config.order, range()) += value / std::abs(config.weight_value);
 }

 // ----------
 void collect_results(mpi::communicator c) {

  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  sn = mpi::reduce(sn_node, c);

  // Computing the average and error values
  for (int k = 0; k < size_n; k++) {
   pn(k) = data_histogram_pn(k);
   if (pn(k) == 0) pn(k) = 1; // Avoids divide by zero, sn(k) should be zero in this case
   sn(k, range()) = sn(k, range()) / pn(k);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   // pn_errors(k) =
   //    (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(k) * (nb_measures -
   //    data_histogram_pn(k)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   // sn_errors(k) = (2 / double(pow(data_histogram_pn(k), 2))) *
   //               sqrt((data_histogram_pn(k) - 1) * data_histogram_sn(k) * (data_histogram_pn(k) -
   //               data_histogram_sn(k)));
   sn_errors(k) = nan("");
   pn_errors(k) = nan("");
  }
 }
};
