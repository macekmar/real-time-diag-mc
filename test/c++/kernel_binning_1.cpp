#include "../c++/binning.hpp"

int main() {
 /// Test the default KernelBinning

 auto binning = KernelBinning(0., 10., 4, 1, false);
 // bins must be ]0; 2.5], ]2.5; 5], ]5; 7.5], ]7.5, 10]
 if (binning.get_bin_length() != 2.5) return 1;

 bool coord_array_ok = true;
 for (int a : {0, 1}) {
  coord_array_ok = coord_array_ok and
                   binning.coord_array(0, a) == keldysh_contour_pt{0, 1.25, a} and
                   binning.coord_array(1, a) == keldysh_contour_pt{0, 3.75, a} and
                   binning.coord_array(2, a) == keldysh_contour_pt{0, 6.25, a} and
                   binning.coord_array(3, a) == keldysh_contour_pt{0, 8.75, a};
 }
 if (not coord_array_ok) return 2;

 binning.add(1, {0, 2.5, 0}, {0., 1.}); // go in bin 0
 binning.add(1, {0, 3., 0}, {1.5, 0.}); // go in bin 1
 binning.add(1, {0, 4., 0}, {0.7, -0.3}); // go in bin 1
 binning.add(1, {0, 10., 0}, {0., 1.1}); // go in bin 3

 // out of range values : these must not be taken into account
 binning.add(1, {0, 0., 0}, {1., 1.});
 binning.add(1, {0, 10.1, 0}, {0.9, 1.});
 binning.add(1, {0, -0.1, 0}, {0.2, 0.});

 auto values = binning.get_values();
 std::cout << values << std::endl;
 bool values_ok = values(0, 0, 0) == dcomplex{0., 1.} and
                  values(0, 1, 0) == dcomplex{2.2, -0.3} and
                  values(0, 2, 0) == dcomplex{0., 0.} and
                  values(0, 3, 0) == dcomplex{0., 1.1};
 if (not values_ok) return 3;

 auto nb_values = binning.get_nb_values();
 std::cout << values << std::endl;
 bool nb_values_ok = nb_values(0, 0, 0) == 1 and
                     nb_values(0, 1, 0) == 2 and
                     nb_values(0, 2, 0) == 0 and
                     nb_values(0, 3, 0) == 1;
 if (not nb_values_ok) return 4;

 std::cout << "success" << std::endl;
 return 0;
}
