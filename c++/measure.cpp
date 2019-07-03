#include "./measure.hpp"

using namespace triqs::statistics;
namespace mpi = triqs::mpi;


Measure::Measure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* weight, array<double, 1>* abs_weight) 
   : config(*config), pn(*pn), weight(*weight), abs_weight(*abs_weight) {
 nb_orders = first_dim(*pn);
 histogram_pn = histogram(0, nb_orders - 1);
 weight_accum = *weight;
 weight_accum() = 0;
 abs_weight_accum = *abs_weight;
 abs_weight_accum() = 0;
};

void Measure::accumulate(dcomplex sign) {
 histogram_pn << config.order;
 weight_accum(config.order) += config.accepted_weight;
 abs_weight_accum(config.order) += std::abs(config.accepted_weight);
}

void Measure::collect_results(mpi::communicator c) {
 // Barrier should be in inherited collect_results !

 // gather pn
 auto data_histogram_pn = histogram_pn.data();

 for (int k = 0; k < nb_orders; k++) {
  pn(k) = data_histogram_pn(k);
 }
 pn = mpi::mpi_all_reduce(pn, c);


 weight = mpi::mpi_all_reduce(weight_accum, c);
 abs_weight = mpi::mpi_all_reduce(abs_weight_accum, c);

 for (int k = 0; k < nb_orders; k++) {
  if (pn(k) != 0) {
   weight(k) = weight(k) / pn(k);
   abs_weight(k) = abs_weight(k) / pn(k);
  }
  else {
   weight(k) = 0;
   abs_weight(k) = 0;
  }
 }
}

// ----------
WeightSignMeasure::WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn, array<dcomplex, 1>* weight, array<double, 1>* abs_weight)
   : Measure(config, pn, weight, abs_weight), sn(*sn) {
 sn_accum = *sn;
 sn_accum() = 0;
}

// ----------
void WeightSignMeasure::accumulate(dcomplex sign) {
 Measure::accumulate(sign);
 dcomplex weight = config.accepted_weight;
 sn_accum(config.order) += weight / std::abs(weight);
}

// ----------
void WeightSignMeasure::collect_results(mpi::communicator c) {
 c.barrier();

 Measure::collect_results(c);

 // gather sn
 sn = mpi::mpi_all_reduce(sn_accum, c);
 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i

 for (int k = 0; k < nb_orders; k++) {
  if (pn(k) != 0)
   sn(k) = sn(k) / pn(k);
  else
   sn(k) = 0;

  sn(k) *= i_n[k % 4];
  weight(k) *= i_n[k % 4];  // This is not done in general since in the kernel method, the weight is already an absolute value and multiplying with i**order is meaningless
 }
}


// ----------
TwoDetKernelMeasure::TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning,
                                         array<long, 1>* pn, array<dcomplex, 4>* kernels,
                                         array<dcomplex, 4>* kernel_diracs, array<long, 4>* nb_kernels,
                                         array<dcomplex, 1>* weight, array<double, 1>* abs_weight)
   : Measure(config, pn, weight, abs_weight),
     kernels_binning(*kernels_binning),
     kernels(*kernels),
     kernel_diracs(*kernel_diracs),
     nb_kernels(*nb_kernels) {};

// ----------
void TwoDetKernelMeasure::accumulate(dcomplex sign) {
 Measure::accumulate(sign);
 config.incr_cycles_trapped();
 
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
                        config.accepted_kernels(p, k_index) / std::abs(config.model_weight));
    p++;
   }
   while (p < config.matrices[0].size()) {
    alpha = config.matrices[0].get_y(p);
    alpha.k_index = k_index;
    kernels_binning.add_dirac(config.order, alpha,
                              config.accepted_kernels(p, k_index) / std::abs(config.model_weight));
    p++;
   }
  }
 }
}

// ----------
void TwoDetKernelMeasure::collect_results(mpi::communicator c) {
 c.barrier();

 Measure::collect_results(c);

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
