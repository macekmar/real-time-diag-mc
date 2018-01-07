#pragma once
#include "./configuration.hpp"
#include "./binning.hpp"
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

using namespace triqs::statistics;
namespace mpi = triqs::mpi;

// -----------------------

class WeightSignMeasure {

 private:
 Configuration& config;
 array<long, 1>& pn;
 array<dcomplex, 3>& sn;
 array<dcomplex, 3> sn_accum;
 int nb_orders;
 histogram histogram_pn;

 public:
 WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 3>* sn);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};

// -----------------------

class TwoDetKernelMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels_binning;
 array<long, 1>& pn;
 array<dcomplex, 3>& kernels;
 array<dcomplex, 3>& kernel_diracs;
 array<long, 3>& nb_kernels;
 int nb_orders;
 histogram histogram_pn;

 public:
 TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning, array<long, 1>* pn,
                     array<dcomplex, 3>* kernels, array<dcomplex, 3>* kernel_diracs,
                     array<long, 3>* nb_kernels);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};
