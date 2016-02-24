#pragma once
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/clef.hpp>
#include <triqs/arrays.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
namespace mpi = triqs::mpi;
using namespace triqs::statistics;

struct measure_pn_sn {

 qmc_data_t* data; // Pointer to the MC qmc_data_t
 array<double, 1> pn;
 array<double, 1> sn;
 array<double, 1> pn_errors;
 array<double, 1> sn_errors;
 int size_n;

 // Creating an array of observables
 array<observable<double>, 1> observable_pn;
 array<observable<double>, 1> observable_sn;

 // With the new observable type
 measure_pn_sn(qmc_data_t* data, array<double, 1>* pn, array<double, 1>* sn, array<double, 1>* pn_errors,
               array<double, 1>* sn_errors)
    : data(data), pn(*pn), sn(*sn), pn_errors(*pn_errors), sn_errors(*sn_errors) {
  size_n = first_dim(*pn);
  observable_pn.resize(size_n);
  observable_sn.resize(size_n);
 }

 void accumulate(dcomplex sign) {

  int k = data->perturbation_order;

  // Accumulating in pn
  for (int i = 0; i < size_n; i++) {
   if (i == k)
    observable_pn(i) << 1;
   else
    observable_pn(i) << 0;
  }

  // Accumulating in sn
  observable_sn(k) << real(sign);
 }

 void collect_results(mpi::communicator c) {

  // Putting the values from all cores together
  auto observable_pn_gathered = array<observable<double>, 1>(size_n);
  auto observable_sn_gathered = array<observable<double>, 1>(size_n);
  for (int i = 0; i < size_n; i++) {
   auto&& series_pn = observable_pn(i).get_series();
   auto&& series_sn = observable_sn(i).get_series();
   observable_pn_gathered(i) = observable<double>(mpi_all_gather(series_pn, c));
   observable_sn_gathered(i) = observable<double>(mpi_all_gather(series_sn, c));

  // Now we treat the data to get the correct average values and errors.
   auto aver_and_err_pn = average_and_error(observable_pn_gathered(i));
   auto aver_and_err_sn = average_and_error(observable_sn_gathered(i));
   pn(i) = aver_and_err_pn.value;
   sn(i) = aver_and_err_sn.value;
   pn_errors(i) = aver_and_err_pn.error_bar;
   sn_errors(i) = aver_and_err_sn.error_bar;
  }
 }
};
