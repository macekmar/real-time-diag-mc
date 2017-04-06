#include "./measure.hpp"

using namespace triqs::arrays;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;


// ----------
WeightSignMeasure::WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<long, 1>* pn_all,
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
                                         array<long, 1>* pn, array<long, 1>* pn_all, array<dcomplex, 3>* sn,
                                         array<dcomplex, 3>* sn_all,
                                         const array<keldysh_contour_pt, 2>* tau_array,
                                         const array<dcomplex, 2>* g0_array, g0_keldysh_t green_function,
                                         const double delta_t)
   : config(*config),
     kernels_binning(*kernels_binning),
     pn(*pn),
     pn_all(*pn_all),
     sn(*sn),
     sn_all(*sn_all),
     tau_array(*tau_array),
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

 array<dcomplex, 2> value = g0_array;
 value() = 0;

 if (config.order == 0) {

  value += g0_array;

 } else {

  array<dcomplex, 2> kernels = config.kernels_evaluate_cofact();
  keldysh_contour_pt alpha_p;

  for (int p = 0; p < config.order; ++p) {
   alpha_p = config.get_config(p);
   for (int k_index : {1, 0}) {
    alpha_p = flip_index(alpha_p);
    auto gf_map = map([&](keldysh_contour_pt tau) { return green_function(tau, alpha_p); });
    value += make_matrix(gf_map(tau_array)) * kernels(p, k_index);
   }
  }
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
                                         array<long, 1>* pn, array<long, 1>* pn_all, array<dcomplex, 3>* sn,
                                         array<dcomplex, 3>* sn_all, array<dcomplex, 3>* kernels, array<dcomplex, 3>* kernels_all,
                                         const array<keldysh_contour_pt, 2>* tau_array,
                                         const array<dcomplex, 2>* g0_array, g0_keldysh_t green_function,
                                         const double delta_t)
   : config(*config),
     kernels_binning(*kernels_binning),
     pn(*pn),
     pn_all(*pn_all),
     sn(*sn),
     sn_all(*sn_all),
     kernels(*kernels),
     kernels_all(*kernels_all),
     tau_array(*tau_array),
     g0_array(*g0_array),
     green_function(green_function),
     delta_t(delta_t) {

 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
}

// ----------
void TwoDetKernelMeasure::accumulate(dcomplex sign) {

 histogram_pn << config.order;

 if (config.order == 0) {
  kernels_binning.add(0, config.get_right_input(), 1 / std::abs(config.weight_value));

 } else {
  keldysh_contour_pt alpha_p;

  for (int p = 0; p < config.order; ++p) {
   alpha_p = config.get_config(p);
   for (int k_index : {1, 0}) {
    alpha_p = flip_index(alpha_p);
    kernels_binning.add(config.order, alpha_p, config.accepted_kernels(p, k_index) / std::abs(config.weight_value));
   }
  }
 }
}

// ----------
void TwoDetKernelMeasure::collect_results(mpi::communicator c) {
 MPI_Barrier(MPI_COMM_WORLD);

 // gather pn
 if (c.rank() == 0) std::cout << "Gathering pn..." << std::endl;
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 std::cout << "rank " << c.rank() << ": reducing " << sum(pn) << std::endl;
 pn_all = mpi::mpi_all_reduce(pn, c);

 std::cout << "rank " << c.rank() << ": nb of measures: " << sum(pn) << std::endl;

 // gather kernels
 if (c.rank() == 0) std::cout << "Gathering kernels..." << std::endl;
 kernels = kernels_binning.get_values();

 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 for (int k = 0; k < nb_orders; k++) {
  kernels(k, ellipsis()) *= i_n[k % 4];
  //kernels(k, ellipsis()) *= i_n[k % 4] / (2 * delta_t);
 }

 kernels_all = mpi::mpi_all_reduce(kernels, c);

 for (int k = 0; k < nb_orders; ++k) {
  if (pn(k) != 0) kernels(k, ellipsis()) /= pn(k);
  if (pn_all(k) != 0) kernels_all(k, ellipsis()) /= pn_all(k);
 }

 // compute sn from kernels
 if (c.rank() == 0) std::cout << "Computing sn from kernels..." << std::endl;
 keldysh_contour_pt tau;
 for (int order = 0; order < nb_orders; ++order) { // for each order
  for (int i = 0; i < second_dim(sn); ++i) {       // for each tau (time)
   for (int a = 0; a < third_dim(sn); ++a) {       // for each tau (keldysh index)
    tau = tau_array(i, a);
    auto gf_map = map([&](keldysh_contour_pt alpha) { return green_function(tau, alpha); });
    auto gf_tau_alpha = gf_map(kernels_binning.coord_array());
    sn(order, i, a) = sum(gf_tau_alpha * kernels(order, ellipsis()));
    sn_all(order, i, a) = sum(gf_tau_alpha * kernels_all(order, ellipsis()));
   }
  }
 }
}
