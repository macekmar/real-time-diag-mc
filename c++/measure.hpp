#pragma once
#include "./configuration.hpp"
#include "./binning.hpp"
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

// -----------------------

template <typename Conf>
class WeightSignMeasure {

 private:
 Conf& config;
 array<long, 1>& pn;
 array<dcomplex, 1>& sn;
 array<dcomplex, 1> sn_accum;
 int nb_orders;
 triqs::statistics::histogram histogram_pn;

 public:
 WeightSignMeasure(Conf* config, array<long, 1>* pn, array<dcomplex, 1>* sn);

 void accumulate(dcomplex sign);

 void collect_results(triqs::mpi::communicator c);
};

// -----------------------
template <typename Conf>
class TwoDetKernelMeasure {

 private:
 Conf& config;
 KernelBinning& kernels_binning;
 array<long, 1>& pn;
 array<dcomplex, 4>& kernels;
 array<dcomplex, 4>& kernel_diracs;
 array<long, 4>& nb_kernels;
 int nb_orders;
 triqs::statistics::histogram histogram_pn;

 public:
 TwoDetKernelMeasure(Conf* config, KernelBinning* kernels_binning, array<long, 1>* pn,
                     array<dcomplex, 4>* kernels, array<dcomplex, 4>* kernel_diracs,
                     array<long, 4>* nb_kernels);

 void accumulate(dcomplex sign);

 void collect_results(triqs::mpi::communicator c);
};
