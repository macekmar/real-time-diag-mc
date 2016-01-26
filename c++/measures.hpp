#pragma once
#include <triqs/mc_tools.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

struct measure_cs {

 qmc_data_t const *data; // Pointer to the MC qmc_data_t
 array<double, 2> &cn_sn;
 long Z = 0;

 measure_cs(qmc_data_t const *data, array<double, 2> &cn_sn) : data(data), cn_sn(cn_sn) { cn_sn() = 0; }

 void accumulate(dcomplex sign) {
  Z++;
  int N = data->perturbation_order;
  cn_sn(0, N) += 1;
  cn_sn(1, N) += real(sign);
 }

 void collect_results(mpi::communicator c) {
  Z = mpi_all_reduce(Z, c);
  cn_sn = mpi_all_reduce(cn_sn, c);
  for (int i = 0; i < second_dim(cn_sn()); ++i) {
   if (std::isnormal(cn_sn(0, i))) cn_sn(1, i) /= cn_sn(0, i);
   cn_sn(0, i) /= Z;
  }
 }
};

