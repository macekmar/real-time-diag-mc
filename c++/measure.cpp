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
 dcomplex weight = config.accepted_weight;
 sn_accum(config.order, ellipsis()) += weight / std::abs(weight);
}

// ----------
void WeightSignMeasure::collect_results(mpi::communicator c) {
 std::cout << "Waiting for other processes before gathering..." << std::endl;
 MPI_Barrier(MPI_COMM_WORLD);
 std::cout << "Gathering..." << std::endl;

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
TwoDetKernelMeasure::TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning,
                                         array<long, 1>* pn, array<long, 1>* pn_all,
                                         array<dcomplex, 3>* kernels, array<dcomplex, 3>* kernels_all)
   : config(*config),
     kernels_binning(*kernels_binning),
     pn(*pn),
     pn_all(*pn_all),
     kernels(*kernels),
     kernels_all(*kernels_all) {

 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
}

// ----------
void TwoDetKernelMeasure::accumulate(dcomplex sign) {

 histogram_pn << config.order;

 if (config.order == 0) {
  return;

 } else {
  keldysh_contour_pt alpha_p;

  for (int p = 0; p < config.matrices[0].size(); ++p) {
   alpha_p = config.get_config(p);
   for (int k_index : {1, 0}) {
    alpha_p = flip_index(alpha_p);
    kernels_binning.add(config.order, alpha_p,
                        config.accepted_kernels(p, k_index) / std::abs(config.accepted_weight));
   }
  }
 }
}

// ----------
void TwoDetKernelMeasure::collect_results(mpi::communicator c) {
 std::cout << "Waiting for other processes before gathering..." << std::endl;
 MPI_Barrier(MPI_COMM_WORLD);
 std::cout << "Gathering..." << std::endl;

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn_all = mpi::mpi_all_reduce(pn, c);

 // gather kernels
 kernels = kernels_binning.get_values();

 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 for (int k = 1; k < nb_orders; k++) {
  kernels(k - 1, ellipsis()) *= i_n[k % 4];
 }

 kernels_all = mpi::mpi_all_reduce(kernels, c);

 for (int k = 1; k < nb_orders; ++k) {
  if (pn(k) != 0) kernels(k - 1, ellipsis()) /= pn(k);
  if (pn_all(k) != 0) kernels_all(k - 1, ellipsis()) /= pn_all(k);
 }
}
