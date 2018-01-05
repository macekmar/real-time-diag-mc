#include "../c++/measure.hpp"

int main() {
 /// Test Dirac deltas in KernelBinning

 auto binning = KernelBinning(0., 10., 4, 2, false);

 binning.add_dirac(1, {0, -2.5, 0}, {0., 1.});
 binning.add_dirac(1, {0, 3., 0}, {1.5, 0.});
 binning.add_dirac(1, {0, 3., 0}, {0.7, -0.3});
 binning.add_dirac(2, {0, 7., 0}, {0., 1.1});

 auto values = binning.get_dirac_values();
 auto times = binning.get_dirac_times();
 std::cout << values << std::endl;
 std::cout << times << std::endl;

 if (first_dim(values) != 2 or second_dim(values) != 3 or third_dim(values) != 2) return 1;

 auto ref_values = array<dcomplex, 3>(2, 3, 2);
 ref_values() = 0;
 ref_values(0, 0, 0) = {0, 1};
 ref_values(0, 1, 0) = {2.2, -0.3};
 ref_values(1, 2, 0) = {0, 1.1};
 auto ref_times = array<double, 1>{-2.5, 3., 7.};

 if (values != ref_values) return 2;
 if (times != ref_times) return 3;

 binning.add_dirac(1, {0, 3., 1}, {0.5, 0.});
 binning.add_dirac(1, {0, 7., 0}, {0.6, 0.});
 binning.add_dirac(1, {0, 7.1, 0}, {0., 10.});

 values = binning.get_dirac_values();
 times = binning.get_dirac_times();
 std::cout << values << std::endl;
 std::cout << times << std::endl;

 if (first_dim(values) != 2 or second_dim(values) != 4 or third_dim(values) != 2) return 4;

 ref_values = array<dcomplex, 3>(2, 4, 2);
 ref_values() = 0;
 ref_values(0, 0, 0) = {0, 1};
 ref_values(0, 1, 0) = {2.2, -0.3};
 ref_values(1, 2, 0) = {0, 1.1};
 ref_values(0, 1, 1) = {0.5, 0};
 ref_values(0, 2, 0) = {0.6, 0};
 ref_values(0, 3, 0) = {0, 10};
 ref_times = array<double, 1>{-2.5, 3., 7., 7.1};

 if (values != ref_values) return 5;
 if (times != ref_times) return 6;

 std::cout << "success" << std::endl;
 return 0;
}
