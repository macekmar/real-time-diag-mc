#pragma once
#include "./qmc_data.hpp"


class KernelBinning {

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
 KernelBinning(double t_min_, double t_max_, int nb_bins_, int max_order, bool match_boundaries);

 void add(int order, keldysh_contour_pt alpha, dcomplex value);
 void add_dirac(int order, keldysh_contour_pt alpha, dcomplex value);

 array<dcomplex, 3> get_values() const { return values; };   // copy
 array<long, 3> get_nb_values() const { return nb_values; }; // copy
 double get_bin_length() const { return bin_length; };
 array<double, 1> get_bin_times() const { return bin_times; };

 array<dcomplex, 3> get_dirac_values() const;
 array<double, 1> get_dirac_times() const;

 // array_const_view<keldysh_contour_pt, 2> get_coord_array() const { return coord_array(); }; // view
 // doesnt work ??
};

