#include "../c++/binning.hpp"
#include "./utility.hpp"

#define ABS_TOL 1.e-14
#define REL_TOL 1.e-10

int main() {
 /// Test the default KernelBinning

 auto binning = KernelBinning(0., 10., 4, 1, 1, false);
 // bins must be ]0; 2.5], ]2.5; 5], ]5; 7.5], ]7.5, 10]
 if (binning.get_bin_length() != 2.5) return 1;

 auto bin_times = binning.get_bin_times();
 auto bin_times_ref = array<double, 1>{1.25, 3.75, 6.25, 8.75};
 if (not is_close_array(bin_times, bin_times_ref, REL_TOL, ABS_TOL)) return 2;

 binning.add(1, {0, up, 2.5, 0}, {0., 1.}); // go in bin 0
 binning.add(1, {0, up, 3., 0}, {1.5, 0.}); // go in bin 1
 binning.add(1, {0, up, 4., 0}, {0.7, -0.3}); // go in bin 1
 binning.add(1, {0, up, 10., 0}, {0., 1.1}); // go in bin 3

 // out of range values : these must not be taken into account
 binning.add(1, {0, up, 0., 0}, {1., 1.});
 binning.add(1, {0, up, 10.1, 0}, {0.9, 1.});
 binning.add(1, {0, up, -0.1, 0}, {0.2, 0.});

 auto values = binning.get_values();
 std::cout << values << std::endl;
 bool values_ok = values(0, 0, 0, 0) == dcomplex{0., 1.} and
                  values(0, 1, 0, 0) == dcomplex{2.2, -0.3} and
                  values(0, 2, 0, 0) == dcomplex{0., 0.} and
                  values(0, 3, 0, 0) == dcomplex{0., 1.1};
 if (not values_ok) return 3;

 auto nb_values = binning.get_nb_values();
 std::cout << values << std::endl;
 bool nb_values_ok = nb_values(0, 0, 0, 0) == 1 and
                     nb_values(0, 1, 0, 0) == 2 and
                     nb_values(0, 2, 0, 0) == 0 and
                     nb_values(0, 3, 0, 0) == 1;
 if (not nb_values_ok) return 4;

 std::cout << "success" << std::endl;
 return 0;
}
