#include "./binning.hpp"

// ----------
// each bin is ]t_min + n*bin_length; t_min + (n+1)*bin_length], for n from 0 to nb_bins-1
// t_max must be included, as it is probably a special time.
KernelBinning::KernelBinning(double t_min_, double t_max_, int nb_bins_, int max_order, int nb_orbitals, bool match_boundaries = false)
    : t_max(t_max_), t_min(t_min_), bin_length((t_max_ - t_min_) / nb_bins_), nb_bins(nb_bins_) {

 if (match_boundaries) {
  double delta_t = t_max - t_min;
  t_min -= 0.5 * delta_t / (nb_bins - 1);
  t_max += 0.5 * delta_t / (nb_bins - 1);
  bin_length = (t_max - t_min) / nb_bins;
 }

 values = array<dcomplex, 4>(max_order, nb_bins, 2, nb_orbitals); // from order 1 to order max_order
 values() = 0;
 nb_values = array<long, 4>(max_order, nb_bins, 2, nb_orbitals);
 nb_values() = 0;

 bin_times = array<double, 1>(nb_bins);
 double time = t_min + 0.5 * bin_length; // middle of the bin
 for (int i = 0; i < nb_bins; ++i) {
  bin_times(i) = time;
  time += bin_length;
 }
}

// ----------
void KernelBinning::add(int order, keldysh_contour_pt alpha, dcomplex value) {
 bool in_range = t_min < alpha.t and alpha.t <= t_max and 0 < order and order <= first_dim(values);
 if (in_range) {
  int bin = int(ceil((alpha.t - t_min) / bin_length)) - 1;
  values(order - 1, bin, alpha.k_index, alpha.x) += value;
  nb_values(order - 1, bin, alpha.k_index, alpha.x)++;
 }
}

// ----------
void KernelBinning::add_dirac(int order, keldysh_contour_pt alpha, dcomplex value) {
 // if not already in the map, create a new dirac
 auto res = diracs.emplace(std::piecewise_construct, std::make_tuple(alpha.t),
                           std::make_tuple(first_dim(values), 2, fourth_dim(values)));

 // if just created, initialize the dirac to zero
 if (res.second) res.first->second() = 0;

 // add the value to the dirac
 res.first->second(order - 1, alpha.k_index, alpha.x) += value;
}

// ----------
array<dcomplex, 4> KernelBinning::get_dirac_values() const {
 auto output = array<dcomplex, 4>(first_dim(values), diracs.size(), 2, fourth_dim(values));
 size_t i = 0;
 for (auto it = diracs.begin(); it != diracs.end(); it++) {
  output(range(), i, range(), range()) = it->second;
  i++;
 }
 return output;
}

// ----------
array<double, 1> KernelBinning::get_dirac_times() const {
 auto output = array<double, 1>(diracs.size());
 size_t i = 0;
 for (auto it = diracs.begin(); it != diracs.end(); it++) {
  output(i) = it->first;
  i++;
 }
 return output;
}

