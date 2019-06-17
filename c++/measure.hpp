#pragma once
#include "./configuration.hpp"
#include "./binning.hpp"
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

// -----------------------
class Measure {
 protected:
  Configuration& config;
  array<long, 1>& pn;
  int nb_orders;
  triqs::statistics::histogram histogram_pn;

 public:
  Measure(Configuration* config, array<long, 1>* pn);

  virtual void accumulate(dcomplex sign) {};
  virtual void collect_results(triqs::mpi::communicator c) {};

};

// -----------------------

class WeightSignMeasure : public Measure {
 protected:
 array<dcomplex, 1>& sn;
 array<dcomplex, 1> sn_accum;
 

 public:
  WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn);
  virtual void accumulate(dcomplex sign) ;
  void collect_results(triqs::mpi::communicator c) ;
};

class WeightMeasure : public WeightSignMeasure {
 public:
  void accumulate(dcomplex sign) ;
  WeightMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 1>* sn);
};

// -----------------------

class TwoDetKernelMeasure : public Measure {
 private:
  KernelBinning& kernels_binning;
  array<dcomplex, 4>& kernels;
  array<dcomplex, 4>& kernel_diracs;
  array<long, 4>& nb_kernels;
 
 public:
  TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning, array<long, 1>* pn,
                      array<dcomplex, 4>* kernels, array<dcomplex, 4>* kernel_diracs,
                      array<long, 4>* nb_kernels);
  void accumulate(dcomplex sign) ;
  void collect_results(triqs::mpi::communicator c) ;
};
