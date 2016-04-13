#pragma once
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>
#include <triqs/clef.hpp>
#include <triqs/arrays.hpp>
#include "./qmc_data.hpp"

using namespace triqs::gfs;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;

struct measure_pn_sn {

 qmc_data_t* data; // Pointer to the MC qmc_data_t
 array<double, 1>& pn;
 array<double, 1>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;

 // Creating an histogram
 histogram histogram_pn;
 histogram histogram_sn;

 // ----------
 measure_pn_sn(qmc_data_t* data, array<double, 1>* pn, array<double, 1>* sn, array<double, 1>* pn_errors,
               array<double, 1>* sn_errors)
    : data(data), pn(*pn), sn(*sn), pn_errors(*pn_errors), sn_errors(*sn_errors) {
  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  histogram_sn = histogram(0, size_n - 1);
 }

 // ----------
 void accumulate(dcomplex sign) {

  int k = data->perturbation_order;
  histogram_pn << k;
  // we only keep the number of + signs, then the number of - signs is just pn(i) - (number of + signs)
  if (real(sign) > 0) histogram_sn << k;
 }

 // ----------
 void collect_results(mpi::communicator c) {

  // We do it with the histogram class
  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  histogram histogram_sn_full = mpi_reduce(histogram_sn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  auto data_histogram_sn = histogram_sn_full.data();
  int n_points = histogram_pn_full.n_data_pts();
  // Computing the average and error values
  // For pn
  for (int i = 0; i < size_n; i++) {
   pn(i) = data_histogram_pn(i) / n_points; // Average
   sn(i) = 2 * (data_histogram_sn(i) / (data_histogram_pn(i))) - 1;

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   pn_errors(i) =
       (1 / double(pow(n_points, 2))) * sqrt((n_points - 1) * data_histogram_pn(i) * (n_points - data_histogram_pn(i)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   sn_errors(i) = (2 / double(pow(data_histogram_pn(i), 2))) *
                  sqrt((data_histogram_pn(i) - 1) * data_histogram_sn(i) * (data_histogram_pn(i) - data_histogram_sn(i)));
  }
 }
};
