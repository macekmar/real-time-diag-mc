#include "../c++/binning.hpp"

int main() {
 /// Test Dirac deltas in KernelBinning

 auto binning = KernelBinning(0., 10., 4, 2, 1, false);

 binning.add_dirac(1, {0, up, -2.5, 0}, {0., 1.}); // should be included even if it's out of window
 binning.add_dirac(1, {0, up, 3., 0}, {1.5, 0.});
 binning.add_dirac(1, {0, up, 3., 0}, {0.7, -0.3});
 binning.add_dirac(2, {0, up, 7., 0}, {0., 1.1});

 auto values = binning.get_dirac_values();
 auto times = binning.get_dirac_times();
 std::cout << values << std::endl;
 std::cout << times << std::endl;

 if (first_dim(values) != 2) return 10; // order
 if (second_dim(values) != 4) return 11; // time
 if (third_dim(values) != 2) return 12; // keldysh index
 if (fourth_dim(values) != 1) return 13; // orbital

 auto ref_values = array<dcomplex, 4>(2, 4, 2, 1);
 ref_values() = 0;
 ref_values(0, 0, 0, 0) = {0, 1}; // t=-2.5
 ref_values(0, 1, 0, 0) = {0, 0}; // t=0
 ref_values(0, 2, 0, 0) = {2.2, -0.3}; // t=3
 ref_values(1, 3, 0, 0) = {0, 1.1}; // t=7
 auto ref_times = array<double, 1>{-2.5, 0., 3., 7.};

 if (values != ref_values) return 2;
 if (times != ref_times) return 3;

 binning.add_dirac(1, {0, up, 3., 1}, {0.5, 0.});
 binning.add_dirac(1, {0, up, 7., 0}, {0.6, 0.});
 binning.add_dirac(1, {0, up, 7.1, 0}, {0., 10.});

 values = binning.get_dirac_values();
 times = binning.get_dirac_times();
 std::cout << values << std::endl;
 std::cout << times << std::endl;

 if (first_dim(values) != 2) return 40;
 if (second_dim(values) != 5) return 41;
 if (third_dim(values) != 2) return 42;
 if (fourth_dim(values) != 1) return 43;

 ref_values = array<dcomplex, 4>(2, 5, 2, 1);
 ref_values() = 0;
 ref_values(0, 0, 0, 0) = {0, 1}; // t=-2.5
 ref_values(0, 1, 0, 0) = {0, 0}; // t=0
 ref_values(0, 2, 0, 0) = {2.2, -0.3}; // t=3
 ref_values(1, 3, 0, 0) = {0, 1.1}; // t=7
 ref_values(0, 2, 1, 0) = {0.5, 0}; // t=3
 ref_values(0, 3, 0, 0) = {0.6, 0}; // t=7
 ref_values(0, 4, 0, 0) = {0, 10}; // t=7.1
 ref_times = array<double, 1>{-2.5, 0., 3., 7., 7.1};

 if (values != ref_values) return 5;
 if (times != ref_times) return 6;

 std::cout << "success" << std::endl;
 return 0;
}
