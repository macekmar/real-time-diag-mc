#include "../c++/measure.hpp"

int main() {
 auto binning = KernelBinning(0., 6., 4, 1, true);
 // bins must be [-1; 1[, [1; 3[, [3; 5[, [5; 7[
 if (binning.get_bin_length() != 2.) return 1;

 bool coord_array_ok = true;
 for (int a : {0, 1}) {
  coord_array_ok = coord_array_ok and
                   binning.coord_array(0, a) == keldysh_contour_pt{0, 0., a} and
                   binning.coord_array(1, a) == keldysh_contour_pt{0, 2., a} and
                   binning.coord_array(2, a) == keldysh_contour_pt{0, 4., a} and
                   binning.coord_array(3, a) == keldysh_contour_pt{0, 6., a};
 }
 if (not coord_array_ok) return 2;

 binning.add(1, {0, -1., 0}, {1., 1.}); // go in bin 0
 binning.add(1, {0, 1., 0}, {0., 1.}); // go in bin 1
 binning.add(1, {0, 2., 0}, {1.5, 0.}); // go in bin 1
 binning.add(1, {0, 2.5, 0}, {0.7, -0.3}); // go in bin 1

 // out of range values : these must not be taken into account
 binning.add(1, {0, 7., 0}, {0., 1.1});
 binning.add(1, {0, 7.1, 0}, {0.9, 1.});
 binning.add(1, {0, -1.1, 0}, {0.2, 0.});

 auto values = binning.get_values();
 bool values_ok = values(0, 0, 0) == dcomplex{1., 1.} and
                  values(0, 1, 0) == dcomplex{2.2, 0.7} and
                  values(0, 2, 0) == dcomplex{0., 0.} and
                  values(0, 3, 0) == dcomplex{0., 0.};
 if (not values_ok) return 3;

 auto nb_values = binning.get_nb_values();
 bool nb_values_ok = nb_values(0, 0, 0) == 1 and
                     nb_values(0, 1, 0) == 3 and
                     nb_values(0, 2, 0) == 0 and
                     nb_values(0, 3, 0) == 0;
 if (not nb_values_ok) return 4;

 return 0;
}
