#pragma once
#include "./measure.hpp"
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

using namespace triqs::gfs;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;

// --------------- Accumulator ----------------
// Implements the measure concept for triqs/mc_generic


class Accumulator {

 private:
 std::shared_ptr<Integrand> integrand;
 array<int, 1>& pn;
 array<dcomplex, 2>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;
 histogram histogram_pn;
 array<dcomplex, 2> sn_node;

 public:
 // ----------
 Accumulator(std::shared_ptr<Integrand> integrand, array<int, 1>* pn, array<dcomplex, 2>* sn, array<double, 1>* pn_errors,
             array<double, 1>* sn_errors, int* nb_measures)
    : integrand(integrand),
      pn(*pn),
      sn(*sn),
      pn_errors(*pn_errors),
      sn_errors(*sn_errors),
      nb_measures(*nb_measures),
      sn_node(*sn) {

  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  sn_node() = 0;
 }

 // ----------
 void accumulate(dcomplex sign) {
  integrand->measure->evaluate();

  histogram_pn << integrand->perturbation_order;
  // FIXME: first dim of sn_node is out of range if min_perturbation_order is not zero !
  sn_node(integrand->perturbation_order, range()) += integrand->measure->get_value() / std::abs(integrand->weight->value);
 }

 // ----------
 void collect_results(mpi::communicator c) {

  // We do it with the histogram class
  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  sn = mpi::reduce(sn_node, c);

  // Computing the average and error values
  for (int k = 0; k < size_n; k++) {
   pn(k) = data_histogram_pn(k);
   if (pn(k) == 0) pn(k) = 1; // Avoids divide by zero, sn(k) should be zero in this case
   sn(k, range()) = sn(k, range()) / pn(k);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   // pn_errors(k) =
   //    (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(k) * (nb_measures -
   //    data_histogram_pn(k)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   // sn_errors(k) = (2 / double(pow(data_histogram_pn(k), 2))) *
   //               sqrt((data_histogram_pn(k) - 1) * data_histogram_sn(k) * (data_histogram_pn(k) - data_histogram_sn(k)));
   sn_errors(k) = nan("");
   pn_errors(k) = nan("");
  }
 }
};
