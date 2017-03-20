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
 // each bin is [t_min + n*bin_length; t_min + (n+1)*bin_length[, for n from 0 to nb_bins-1

 private:
 array<dcomplex, 3> values; // 3D: order, binning, keldysh index
 array<int, 3> nb_values;   // 3D: order, binning, keldysh index
 double t_min, t_max, bin_length;
 int nb_bins;

 public:
 array<keldysh_contour_pt, 2> coord_array;
 KernelBinning(){};
 KernelBinning(double t_min, double t_max, int nb_bins, int max_order, bool match_boundaries = false)
    : t_max(t_max), t_min(t_min), bin_length((t_max - t_min) / nb_bins), nb_bins(nb_bins) {

  if (match_boundaries) {
   double delta_t = t_max - t_min;
   t_min -= 0.5 * delta_t / (nb_bins - 1);
   t_max += 0.5 * delta_t / (nb_bins - 1);
   bin_length = (t_max - t_min) / nb_bins;
  }

  values = array<dcomplex, 3>(max_order + 1, nb_bins, 2); // from order 0 to order max_order
  values() = 0;
  nb_values = array<int, 3>(max_order + 1, nb_bins, 2);
  nb_values() = 0;

  coord_array = array<keldysh_contour_pt, 2>(nb_bins, 2);
  double time = t_min + 0.5 * bin_length; // middle of the bin
  for (int i = 0; i < nb_bins; ++i) {
   if (std::abs(time) <
       1e-6)  // temp fix to make sure a bin match exactly with t'=0 (if match_bundary is true)
    time = 0; // FIXME: do something cleaner
   for (int k_index : {0, 1}) {
    coord_array(i, k_index) = {0, time, k_index};
   }
   time += bin_length;
  }
 };

 void add(int order, keldysh_contour_pt alpha, dcomplex value) {
  assert(t_min <= alpha.t and alpha.t <= t_max);
  int bin = int((alpha.t - t_min) / bin_length);
  bin = std::min(bin, nb_bins - 1); // avoids leak of values for alpha.t == t_max
  values(order, bin, alpha.k_index) += value;
  nb_values(order, bin, alpha.k_index)++;
 };

 array<dcomplex, 3> get_values() const { return values; };  // copy
 array<int, 3> get_nb_values() const { return nb_values; }; // copy

 // array_const_view<keldysh_contour_pt, 2> get_coord_array() const { return coord_array(); }; // view
 // doesnt work ??
};

// -----------------------

class WeightSignMeasure {

 private:
 Configuration& config;
 array<int, 1>& pn;
 array<int, 1>& pn_all;
 array<dcomplex, 3>& sn;
 array<dcomplex, 3>& sn_all;
 array<dcomplex, 3> sn_accum;
 int nb_orders;
 histogram histogram_pn;

 public:
 WeightSignMeasure(Configuration* config, array<int, 1>* pn, array<int, 1>* pn_all, array<dcomplex, 3>* sn,
                   array<dcomplex, 3>* sn_all);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};

// -----------------------

class TwoDetCofactMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels_binning;
 array<int, 1>& pn;
 array<int, 1>& pn_all;
 array<dcomplex, 3>& sn;
 array<dcomplex, 3>& sn_all;
 array<dcomplex, 3> sn_accum;
 int nb_orders;
 histogram histogram_pn;
 const array<keldysh_contour_pt, 2>& tau_array;
 const keldysh_contour_pt taup;
 const int op_to_measure_spin;
 const array<dcomplex, 2>& g0_array;
 g0_keldysh_t green_function;
 const double delta_t;

 public:
 TwoDetCofactMeasure(Configuration* config, KernelBinning* kernels_binning, array<int, 1>* pn,
                     array<int, 1>* pn_all, array<dcomplex, 3>* sn, array<dcomplex, 3>* sn_all,
                     const array<keldysh_contour_pt, 2>* tau_array, const keldysh_contour_pt taup,
                     const int op_to_measure_spin, const array<dcomplex, 2>* g0_array,
                     g0_keldysh_t green_function, const double delta_t);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};

// -----------------------

class TwoDetKernelMeasure {

 private:
 Configuration& config;
 KernelBinning& kernels_binning;
 array<int, 1>& pn;
 array<int, 1>& pn_all;
 array<dcomplex, 3>& sn;
 array<dcomplex, 3>& sn_all;
 array<dcomplex, 3>& kernels_all;
 int nb_orders;
 histogram histogram_pn;
 const array<keldysh_contour_pt, 2>& tau_array;
 const keldysh_contour_pt taup;
 const int op_to_measure_spin;
 const array<dcomplex, 2>& g0_array;
 g0_keldysh_t green_function;
 const double delta_t;

 public:
 TwoDetKernelMeasure(Configuration* config, KernelBinning* kernels_binning, array<int, 1>* pn,
                     array<int, 1>* pn_all, array<dcomplex, 3>* sn, array<dcomplex, 3>* sn_all,
                     array<dcomplex, 3>* kernels_all, const array<keldysh_contour_pt, 2>* tau_array,
                     const keldysh_contour_pt taup, const int op_to_measure_spin,
                     const array<dcomplex, 2>* g0_array, g0_keldysh_t green_function, const double delta_t);

 void accumulate(dcomplex sign);

 void collect_results(mpi::communicator c);
};
