#include "./measure.hpp"

using namespace triqs::statistics;
namespace mpi = triqs::mpi;


Measure::Measure(Configuration* config, array<long, 1>* pn) : config(*config), pn(*pn) {
 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
};

// ----------
WeightSignMeasure::WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn)
   : Measure(config, pn), sn(*sn) {
 sn_accum = *sn;
 sn_accum() = 0;
}

// ----------
void WeightSignMeasure::accumulate(dcomplex sign) {
 histogram_pn << config.order;
 dcomplex weight = config.accepted_weight;
 sn_accum(config.order) += weight / std::abs(weight);
}

// ----------
void WeightSignMeasure::collect_results(mpi::communicator c) {
 c.barrier();

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn = mpi::mpi_all_reduce(pn, c);

 // gather sn
 sn = mpi::mpi_all_reduce(sn_accum, c);
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

 for (int k = 0; k < nb_orders; k++) {
  if (pn(k) != 0)
   sn(k) = sn(k) / pn(k);
  else
   sn(k) = 0;

  sn(k) *= i_n[k % 4];
 }
}

// ----------
void WeightMeasure::accumulate(dcomplex sign) {
 histogram_pn << config.order;
 sn_accum(config.order) += config.accepted_weight;
}

WeightMeasure::WeightMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn)
   : WeightSignMeasure(config, pn, sn) {}

// -----------------------

// ----------
TwoDetKernelMeasure::TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning,
                                         array<long, 1>* pn, array<dcomplex, 4>* kernels,
                                         array<dcomplex, 4>* kernel_diracs, array<long, 4>* nb_kernels)
   : Measure(config, pn),
     kernels_binning(*kernels_binning),
     kernels(*kernels),
     kernel_diracs(*kernel_diracs),
     nb_kernels(*nb_kernels) {};

// ----------
void TwoDetKernelMeasure::accumulate(dcomplex sign) {
 config.incr_cycles_trapped();
 histogram_pn << config.order;

 if (config.order == 0) {
  return;

 } else {
  keldysh_contour_pt alpha;

  for (int k_index : {1, 0}) {
   int p = 0;
   while (p < config.order) {
    alpha = config.matrices[0].get_y(p);
    alpha.k_index = k_index;
    kernels_binning.add(config.order, alpha,
                        config.accepted_kernels(p, k_index) / std::abs(config.accepted_weight));
    p++;
   }
   while (p < config.matrices[0].size()) {
    alpha = config.matrices[0].get_y(p);
    alpha.k_index = k_index;
    kernels_binning.add_dirac(config.order, alpha,
                              config.accepted_kernels(p, k_index) / std::abs(config.accepted_weight));
    p++;
   }
  }
 }
}

// ----------
void TwoDetKernelMeasure::collect_results(mpi::communicator c) {
 c.barrier();

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn = mpi::mpi_all_reduce(pn, c);

 // gather kernels
 kernels = kernels_binning.get_values();
 kernel_diracs = kernels_binning.get_dirac_values();

 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 for (int k = 1; k < nb_orders; k++) {
  kernels(k - 1, ellipsis()) *= i_n[k % 4];
  kernel_diracs(k - 1, ellipsis()) *= i_n[k % 4];
 }
 kernels = mpi::mpi_all_reduce(kernels, c);
 kernel_diracs = mpi::mpi_all_reduce(kernel_diracs, c);

 for (int k = 1; k < nb_orders; ++k) {
  if (pn(k) != 0) {
   kernels(k - 1, ellipsis()) /= pn(k);
   kernel_diracs(k - 1, ellipsis()) /= pn(k);
  }
 }

 // gather nb_kernels
 nb_kernels = kernels_binning.get_nb_values();
 nb_kernels = mpi::mpi_all_reduce(nb_kernels, c);
}
