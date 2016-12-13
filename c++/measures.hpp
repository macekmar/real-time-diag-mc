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
 array<dcomplex, 1>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;

 // Creating an histogram
 histogram histogram_pn;
 triqs::arrays::vector<dcomplex> phase_n;

 // ----------
 measure_pn_sn(qmc_data_t* data, array<double, 1>* pn, array<dcomplex, 1>* sn, array<double, 1>* pn_errors,
               array<double, 1>* sn_errors, int* nb_measures)
    : data(data), pn(*pn), sn(*sn), pn_errors(*pn_errors), sn_errors(*sn_errors), nb_measures(*nb_measures) {
  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  phase_n = triqs::arrays::vector<dcomplex>(size_n, 0.0);
 }

 // ----------
 void accumulate(dcomplex phase) {

  int k = data->perturbation_order;
  histogram_pn << k;
  phase_n(k) += phase;
 }

 // ----------
 void collect_results(mpi::communicator c) {

  // We do it with the histogram class
  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  triqs::arrays::vector<dcomplex> phase_n_full = mpi::reduce(phase_n, c);

  // Computing the average and error values
  // For pn
  for (int i = 0; i < size_n; i++) {
   pn(i) = data_histogram_pn(i) / nb_measures; // Average
   //sn(i) = 2 * (data_histogram_sn(i) / (data_histogram_pn(i))) - 1;
   sn(i) = phase_n_full(i) / data_histogram_pn(i);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   pn_errors(i) =
       (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(i) * (nb_measures - data_histogram_pn(i)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   //sn_errors(i) = (2 / double(pow(data_histogram_pn(i), 2))) *
   //               sqrt((data_histogram_pn(i) - 1) * data_histogram_sn(i) * (data_histogram_pn(i) - data_histogram_sn(i)));
   sn_errors(i) = nan("");
  }
 }
};
