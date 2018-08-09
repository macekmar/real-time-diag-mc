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
 array<dcomplex, 1>& sn;
 array<dcomplex, 1> sn_accum;
 int nb_orders;
 histogram histogram_pn;

 public:
 WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};

// -----------------------

class TwoDetKernelMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels_binning;
 array<long, 1>& pn;
 array<dcomplex, 4>& kernels;
 array<dcomplex, 4>& kernel_diracs;
 array<long, 4>& nb_kernels;
 int nb_orders;
 histogram histogram_pn;

 public:
 TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning, array<long, 1>* pn,
                     array<dcomplex, 4>* kernels, array<dcomplex, 4>* kernel_diracs,
                     array<long, 4>* nb_kernels);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};
