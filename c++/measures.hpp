#pragma once
#include <triqs/mc_tools.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;

struct measure_pn_sn {

 qmc_data_t const *data; // Pointer to the MC qmc_data_t
 array<double, 1> &pn;
 array<double, 1> &sn;
 long z = 0;

 measure_pn_sn(qmc_data_t const *data, array<double, 1> *pn, array<double, 1> *sn) : data(data), pn(*pn), sn(*sn) {}

 void accumulate(dcomplex sign) {
  z++;
  int k = data->perturbation_order;
  pn(k) += 1;
  sn(k) += real(sign);
 }

 void collect_results(mpi::communicator c) {
  z = mpi_all_reduce(z, c);
  pn = mpi_all_reduce(pn, c);
  sn = mpi_all_reduce(sn, c);
  for (int i = 0; i < first_dim(pn); ++i) {
   if (std::isnormal(pn(i))) sn(i) /= pn(i);
   pn(i) /= z;
  }
  std::cout << "pn = " << pn << std::endl;
 }
};

