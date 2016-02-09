#pragma once
#include <triqs/mc_tools.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

struct measure_cs {

 qmc_data_t const *data; // Pointer to the MC qmc_data_t
 array<double, 1> &cn;
 array<double, 1> &sn;
 long z = 0;

 measure_cs(qmc_data_t const *data, array<double, 1> *cn, array<double, 1> *sn) : data(data), cn(*cn), sn(*sn) {}

 void accumulate(dcomplex sign) {
  z++;
  int k = data->perturbation_order;
  cn(k) += 1;
  sn(k) += real(sign);
 }

 void collect_results(mpi::communicator c) {
  // FIXME
  //  z = mpi_all_reduce(z, c);
  //  cn = mpi_all_reduce(cn, c);
  //  sn = mpi_all_reduce(sn, c);
  mpi_all_reduce(z, c);
  mpi_all_reduce(cn, c);
  mpi_all_reduce(sn, c);
  for (int i = 0; i < first_dim(cn); ++i) {
   if (std::isnormal(cn(i))) sn(i) /= cn(i);
   cn(i) /= z;
  }
 }
};

