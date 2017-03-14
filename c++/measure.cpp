#include "./measure.hpp"

using namespace triqs::arrays;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;


// ----------
WeightSignMeasure::WeightSignMeasure(Configuration* config, array<int, 1>* pn, array<int, 1>* pn_all,
                                     array<dcomplex, 3>* sn, array<dcomplex, 3>* sn_all)
   : config(*config), pn(*pn), pn_all(*pn_all), sn(*sn), sn_all(*sn_all) {

 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
 sn_accum = *sn;
 sn_accum() = 0;
}

// ----------
void WeightSignMeasure::accumulate(dcomplex sign) {
 histogram_pn << config.order;
 sn_accum(config.order, ellipsis()) += config.weight_value / std::abs(config.weight_value);
}

// ----------
void WeightSignMeasure::collect_results(mpi::communicator c) {

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn_all = mpi::mpi_all_reduce(pn, c);

 // gather sn
 sn_all = mpi::mpi_all_reduce(sn_accum, c);
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

 for (int k = 0; k < nb_orders; k++) {
  if (pn(k) != 0)
   sn(k, ellipsis()) = sn_accum(k, ellipsis()) / pn(k);
  else
   sn(k, ellipsis()) = 0;
  if (pn_all(k) != 0)
   sn_all(k, ellipsis()) /= pn_all(k);
  else
   sn_all(k, ellipsis()) = 0;

  sn(k, ellipsis()) *= i_n[k % 4];
  sn_all(k, ellipsis()) *= i_n[k % 4];
 }
}

// -----------------------

// ----------
TwoDetCofactMeasure::TwoDetCofactMeasure(Configuration* config, KernelBinning* kernels_binning,
                                         array<int, 1>* pn, array<int, 1>* pn_all, array<dcomplex, 3>* sn,
                                         array<dcomplex, 3>* sn_all,
                                         const array<keldysh_contour_pt, 2>* tau_array,
                                         const keldysh_contour_pt taup, const int op_to_measure_spin,
                                         const array<dcomplex, 2>* g0_array, g0_keldysh_t green_function,
                                         const double delta_t)
   : config(*config),
     kernels_binning(*kernels_binning),
     pn(*pn),
     pn_all(*pn_all),
     sn(*sn),
     sn_all(*sn_all),
     tau_array(*tau_array),
     taup(taup),
     op_to_measure_spin(op_to_measure_spin),
     g0_array(*g0_array),
     green_function(green_function),
     delta_t(delta_t) {

 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
 sn_accum = *sn;
 sn_accum() = 0;
}

// ----------
void TwoDetCofactMeasure::accumulate(dcomplex sign) {

 histogram_pn << config.order;

 keldysh_contour_pt alpha_p_right, alpha_tmp;
 auto matrix_0 = &(config.matrices[op_to_measure_spin]);
 auto matrix_1 = &(config.matrices[1 - op_to_measure_spin]);
 dcomplex kernel;
 int signs[2] = {1, -1};
 int n = config.order; // perturbation order
 array<dcomplex, 2> value = g0_array;
 value() = 0;

 if (n == 0) {

  value += g0_array;

 } else {

  value() = 0;
  auto tau_weight = config.get_left_input();
  matrix_0->remove(n, n); // remove tau_w and taup

  alpha_tmp = taup;
  matrix_0->roll_matrix(det_manip<g0_keldysh_t>::RollDirection::Left);

  // TODO: put the cofactor calculation into the Configuration class
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

    auto gf_map = map([&](keldysh_contour_pt tau) { return green_function(tau, alpha_p_right); });
    value += make_matrix(gf_map(tau_array)) * kernel;

    if (k_index == 0) alpha_tmp = alpha_p_right;
   }
  }
  matrix_0->change_col(n - 1, alpha_tmp);

  matrix_0->insert(n, n, tau_weight, taup);
 }

 sn_accum(config.order, ellipsis()) += value / std::abs(config.weight_value);
}

// ----------
void TwoDetCofactMeasure::collect_results(mpi::communicator c) {

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn_all = mpi::mpi_all_reduce(pn, c);

 // gather sn
 sn_all = mpi::mpi_all_reduce(sn_accum, c);
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

 for (int k = 0; k < nb_orders; k++) {
  if (pn(k) != 0)
   sn(k, ellipsis()) = sn_accum(k, ellipsis()) / pn(k);
  else
   sn(k, ellipsis()) = 0;
  if (pn_all(k) != 0)
   sn_all(k, ellipsis()) /= pn_all(k);
  else
   sn_all(k, ellipsis()) = 0;

  sn(k, ellipsis()) *= i_n[k % 4] / (2 * delta_t);
  sn_all(k, ellipsis()) *= i_n[k % 4] / (2 * delta_t);
 }
}

// -----------------------

// ----------
TwoDetKernelMeasure::TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning,
                                         array<int, 1>* pn, array<int, 1>* pn_all, array<dcomplex, 3>* sn,
                                         array<dcomplex, 3>* sn_all, array<dcomplex, 3>* kernels_all,
                                         const array<keldysh_contour_pt, 2>* tau_array,
                                         const keldysh_contour_pt taup, const int op_to_measure_spin,
                                         const array<dcomplex, 2>* g0_array, g0_keldysh_t green_function,
                                         const double delta_t)
   : config(*config),
     kernels_binning(*kernels_binning),
     pn(*pn),
     pn_all(*pn_all),
     sn(*sn),
     sn_all(*sn_all),
     kernels_all(*kernels_all),
     tau_array(*tau_array),
     taup(taup),
     op_to_measure_spin(op_to_measure_spin),
     g0_array(*g0_array),
     green_function(green_function),
     delta_t(delta_t) {

 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
}

// ----------
void TwoDetKernelMeasure::accumulate(dcomplex sign) {

 histogram_pn << config.order; // TODO: pn is already accumulated in kernels_binning

 keldysh_contour_pt alpha_p_right, alpha_tmp;
 auto matrix_0 = &(config.matrices[op_to_measure_spin]);
 auto matrix_1 = &(config.matrices[1 - op_to_measure_spin]);
 dcomplex kernel;
 int signs[2] = {1, -1};
 int n = config.order; // perturbation order

 if (n == 0) {
  kernels_binning.add(0, taup, 1 / std::abs(config.weight_value));

 } else {

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
    // std::cout << "DEBUG: " << alpha_p_right.t << " : " << kernel / std::abs(config.weight_value) <<
    // std::endl;
    kernels_binning.add(n, alpha_p_right, kernel / std::abs(config.weight_value));

    if (k_index == 0) alpha_tmp = alpha_p_right;
   }
  }
  matrix_0->change_col(n - 1, alpha_tmp);

  matrix_0->insert(n, n, tau_weight, taup);
 }
}

// ----------
void TwoDetKernelMeasure::collect_results(mpi::communicator c) {

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn_all = mpi::mpi_all_reduce(pn, c);

 // gather kernels
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 for (int k = 0; k < nb_orders; k++) {
  kernels_binning.values(k, ellipsis()) *= i_n[k % 4] / (2 * delta_t);
 }

 std::cout << "DEBUG: nb_values = " << sum(kernels_binning.nb_values) << std::endl;

 array<dcomplex, 3> kernels = kernels_binning.values / kernels_binning.nb_values;
 foreach (kernels, [&kernels](size_t i, size_t j, size_t k) {
  if (my_isnan(kernels(i, j, k))) kernels(i, j, k) = 0;
 })
  ;

 kernels_all = mpi::mpi_all_reduce(kernels_binning.values, c);
 array<int, 3> nb_values_all = mpi::mpi_all_reduce(kernels_binning.nb_values, c);
 kernels_all /= nb_values_all;
 foreach (kernels_all, [this](size_t i, size_t j, size_t k) {
  if (my_isnan(kernels_all(i, j, k))) kernels_all(i, j, k) = 0;
 })
  ;

 // compute sn from kernels
 keldysh_contour_pt tau;
 for (int order = 0; order < nb_orders; ++order) { // for each order
  for (int i = 0; i < second_dim(sn); ++i) {       // for each tau (time)
   for (int a = 0; a < third_dim(sn); ++a) {       // for each tau (keldysh index)
    tau = tau_array(i, a);
    auto gf_map = map([&](keldysh_contour_pt alpha) { return green_function(tau, alpha); });
    sn(order, i, a) = sum(gf_map(kernels_binning.coord_array) * kernels(order, ellipsis()));
    sn_all(order, i, a) = sum(gf_map(kernels_binning.coord_array) * kernels_all(order, ellipsis()));
   }
  }
 }

 // keldysh_contour_pt alpha_p;
 // keldysh_contour_pt tau;
 // for (int order = 0; order < nb_orders; ++order) { // for each order
 // int flatten_idx = 0;
 // for (int i = 0; i < second_dim(sn); ++i) { // for each tau (time)
 //  for (int a = 0; a < third_dim(sn); ++a) { // for each tau (keldysh index)
 //   if (order == 0)
 //    sn(0, i, a) = (*g0_array)(flatten_idx);
 //   else {
 //    tau = (*tau_array)[flatten_idx];
 //    for (int k_index : {1, 0}) { // for each kernel (keldych index)
 //     double time = kernels.t_min;
 //     for (int j = 0; j < kernels.nb_bins; ++j) { // for each kernel (time)
 //      alpha_p = {0, time, k_index};
 //      sn(order, i, a) += green_function(tau, alpha_p) * kernels.values(order - 1, j, k_index);
 //      time += kernels.bin_length;
 //     }
 //    }
 //   }
 //   flatten_idx++;
 //  }
 // }
 //}

 //// gather sn and kernels
 // sn_all = mpi::mpi_all_reduce(sn, c);
 // kernels_binning.all_reduce(c);

 // for (int k = 1; k < nb_orders; k++) {
 // if (pn(k) != 0) {
 //  sn(k, ellipsis()) /= pn(k);
 //  kernels_binning.values(k - 1, ellipsis()) /= pn(k);
 // } else {
 //  sn(k, ellipsis()) = 0;
 //  kernels_binning.values(k - 1, ellipsis()) = 0;
 // }
 // if (pn_all(k) != 0) {
 //  sn_all(k, ellipsis()) /= pn_all(k);
 //  kernels_binning.values_all(k - 1, ellipsis()) /= pn(k);
 // } else {
 //  sn_all(k, ellipsis()) = 0;
 //  kernels_binning.values_all(k - 1, ellipsis()) = 0;
 // }
 //}
}
