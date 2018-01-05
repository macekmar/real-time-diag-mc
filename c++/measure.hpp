#pragma once
#include "./configuration.hpp"
#include <triqs/arrays.hpp>
#include <triqs/mc_tools.hpp>
#include <triqs/statistics.hpp>
#include <triqs/statistics/histograms.hpp>

// using namespace triqs::gfs;
using namespace triqs::statistics;
namespace mpi = triqs::mpi;


class KernelBinning {
 // each bin is ]t_min + n*bin_length; t_min + (n+1)*bin_length], for n from 0 to nb_bins-1
 // t_max must be included, as it is probably a special time.

 private:
 array<dcomplex, 3> values; // 3D: order, binning, keldysh index
 array<long, 3> nb_values;  // 3D: order, binning, keldysh index
 array<double, 1> bin_times;
 std::map<double, array<dcomplex, 2>> diracs;
 double t_min, t_max, bin_length;
 int nb_bins;

 public:
 array<keldysh_contour_pt, 2> coord_array;
 KernelBinning(){};
 KernelBinning(double t_min_, double t_max_, int nb_bins_, int max_order, bool match_boundaries = false)
    : t_max(t_max_), t_min(t_min_), bin_length((t_max_ - t_min_) / nb_bins_), nb_bins(nb_bins_) {

  if (match_boundaries) {
   double delta_t = t_max - t_min;
   t_min -= 0.5 * delta_t / (nb_bins - 1);
   t_max += 0.5 * delta_t / (nb_bins - 1);
   bin_length = (t_max - t_min) / nb_bins;
  }

  values = array<dcomplex, 3>(max_order, nb_bins, 2); // from order 1 to order max_order
  values() = 0;
  nb_values = array<long, 3>(max_order, nb_bins, 2);
  nb_values() = 0;

  coord_array = array<keldysh_contour_pt, 2>(nb_bins, 2);
  bin_times = array<double, 1>(nb_bins);
  double time = t_min + 0.5 * bin_length; // middle of the bin
  for (int i = 0; i < nb_bins; ++i) {
   bin_times(i) = time;
   for (int k_index : {0, 1}) {
    coord_array(i, k_index) = {0, time, k_index};
   }
   time += bin_length;
  }
 };

 void add(int order, keldysh_contour_pt alpha, dcomplex value) {
  bool in_range = t_min < alpha.t and alpha.t <= t_max and 0 < order and order <= first_dim(values);
  if (in_range) {
   int bin = int(ceil((alpha.t - t_min) / bin_length)) - 1;
   values(order - 1, bin, alpha.k_index) += value;
   nb_values(order - 1, bin, alpha.k_index)++;
  }
 };

 void add_dirac(int order, keldysh_contour_pt alpha, dcomplex value) {
  auto res = diracs.emplace(std::piecewise_construct, std::make_tuple(alpha.t),
                            std::make_tuple(first_dim(values), 2)); // add the dirac if not already in the map
  if (res.second) res.first->second() = 0;
  res.first->second(order - 1, alpha.k_index) += value;
 };

 array<dcomplex, 3> get_values() const { return values; };   // copy
 array<long, 3> get_nb_values() const { return nb_values; }; // copy
 double get_bin_length() const { return bin_length; };
 array<double, 1> get_bin_times() const { return bin_times; };

 array<dcomplex, 3> get_dirac_values() const {
  auto output = array<dcomplex, 3>(first_dim(values), diracs.size(), 2);
  size_t i = 0;
  for (auto it = diracs.begin(); it != diracs.end(); it++) {
   output(range(), i, range()) = it->second;
   i++;
  }
  return output;
 };

 array<double, 1> get_dirac_times() const {
  auto output = array<double, 1>(diracs.size());
  size_t i = 0;
  for (auto it = diracs.begin(); it != diracs.end(); it++) {
   output(i) = it->first;
   i++;
  }
  return output;
 };

 // array_const_view<keldysh_contour_pt, 2> get_coord_array() const { return coord_array(); }; // view
 // doesnt work ??
};

// -----------------------

class WeightSignMeasure {

 private:
 Configuration& config;
 array<long, 1>& pn;
 array<dcomplex, 3>& sn;
 array<dcomplex, 3> sn_accum;
 int nb_orders;
 histogram histogram_pn;

 public:
 WeightSignMeasure(Configuration* config, array<long, 1>* pn, array<dcomplex, 3>* sn);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};

// -----------------------

class TwoDetKernelMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels_binning;
 array<long, 1>& pn;
 array<dcomplex, 3>& kernels;
 array<dcomplex, 3>& kernel_diracs;
 array<long, 3>& nb_kernels;
 int nb_orders;
 histogram histogram_pn;

 public:
 TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning, array<long, 1>* pn,
                     array<dcomplex, 3>* kernels, array<dcomplex, 3>* kernel_diracs,
                     array<long, 3>* nb_kernels);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};
