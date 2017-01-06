#pragma once
#include "./qmc_data.hpp"
#include "./weight.hpp"
#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

using namespace triqs::gfs;
using namespace triqs::statistics;
using triqs::arrays::range;
namespace mpi = triqs::mpi;


// --------------- Measure ----------------

struct qmc_measure {

 qmc_weight* weight;
 int perturbation_order = 0;                    // the current perturbation order
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 array<dcomplex, 1> value;                      // Sum of determinants of the last accepted config

 qmc_measure(const qmc_measure&) = delete;    // non construction-copyable
 void operator=(const qmc_measure&) = delete; // non copyable

 // ----------
 qmc_measure(qmc_weight* weight, double t_max, const solve_parameters_t* params, g0_t* g0_lesser, g0_t* g0_greater)
    : weight(weight) {

  value = array<dcomplex, 1>(1);
  value() = 0;

  for (auto spin : {up, down})
   matrices.emplace_back(g0_keldysh_t{g0_adaptor_t{*g0_lesser}, g0_adaptor_t{*g0_greater}, params->alpha, t_max}, 100);

  for (auto spin : {up, down}) {
   auto const& ops = params->op_to_measure[spin];
   if (ops.size() == 2)
    matrices[spin].insert_at_end(make_keldysh_contour_pt(ops[0], params->measure_times.first),
                                 make_keldysh_contour_pt(ops[1], params->measure_times.second));
  }
 }

 // ----------
 void insert(int k, keldysh_contour_pt pt) {
  for (auto spin : {up, down}) {
   matrices[spin].insert(k, k, pt, pt);
  }
  perturbation_order++;
 }

 void insert2(int k, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
  for (auto spin : {up, down}) {
   matrices[spin].insert2(k, k + 1, k, k + 1, pt1, pt2, pt1, pt2);
  }
  perturbation_order += 2;
 }

 void remove(int k) {
  for (auto spin : {up, down}) {
   matrices[spin].remove(k, k);
  }
  perturbation_order--;
 }

 void remove2(int k1, int k2) {
  for (auto spin : {up, down}) {
   matrices[spin].remove2(k1, k2, k1, k2);
  }
  perturbation_order -= 2;
 }

 // ----------
 void evaluate() {
  value(0) = recompute_sum_keldysh_indices(matrices[up], matrices[down], perturbation_order) / std::abs(weight->value);
 }
};


// --------------- Accumulator ----------------
// Implements the measure concept for triqs/mc_generic

struct qmc_accumulator {

 qmc_measure* measure;
 array<double, 1>& pn;
 array<dcomplex, 1>& sn;
 array<double, 1>& pn_errors;
 array<double, 1>& sn_errors;
 int size_n;
 int& nb_measures;
 histogram histogram_pn;
 array<dcomplex, 1> sign_n;

 // ----------
 qmc_accumulator(qmc_measure* measure, array<double, 1>* pn, array<dcomplex, 1>* sn, array<double, 1>* pn_errors,
                 array<double, 1>* sn_errors, int* nb_measures)
    : measure(measure), pn(*pn), sn(*sn), pn_errors(*pn_errors), sn_errors(*sn_errors), nb_measures(*nb_measures) {

  size_n = first_dim(*pn);
  histogram_pn = histogram(0, size_n - 1);
  sign_n = array<dcomplex, 1>(size_n);
  sign_n() = 0;
 }

 // ----------
 void accumulate(dcomplex sign) {
  measure->evaluate();

  histogram_pn << measure->perturbation_order;
  sign_n(measure->perturbation_order) += measure->value(0);
 }

 // ----------
 void collect_results(mpi::communicator c) {

  // We do it with the histogram class
  histogram histogram_pn_full = mpi_reduce(histogram_pn, c);
  auto data_histogram_pn = histogram_pn_full.data();
  nb_measures = histogram_pn_full.n_data_pts();

  array<dcomplex, 1> sign_n_full = mpi::reduce(sign_n, c);

  // Computing the average and error values
  for (int k = 0; k < size_n; k++) {
   pn(k) = data_histogram_pn(k) / nb_measures; // Average
   sn(k) = sign_n_full(k) / data_histogram_pn(k);

   // FIXME : explicit formula for the error bar jacknife of a series of 0 and 1
   pn_errors(k) =
       (1 / double(pow(nb_measures, 2))) * sqrt((nb_measures - 1) * data_histogram_pn(k) * (nb_measures - data_histogram_pn(k)));

   // FIXME : explicit formula as well for the error bar of the sn using a jacknife
   // sn_errors(k) = (2 / double(pow(data_histogram_pn(k), 2))) *
   //               sqrt((data_histogram_pn(k) - 1) * data_histogram_sn(k) * (data_histogram_pn(k) - data_histogram_sn(k)));
   sn_errors(k) = nan("");
  }
 }
};
