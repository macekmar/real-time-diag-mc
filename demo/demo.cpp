#include <mpi.h>
#include "../c++/solver_core.hpp"
#include "../c++/g0_flat_band.hpp"


int main() {

 std::cout << "Hello" << std::endl;
 MPI::Init();

 auto g0 = make_g0_flat_band(20.0, // beta
                             0.2,   // Gamma
                             250.,  // tmax_gf0
                             100,   // Nt_gf0
                             0.0,   // epsilon_d
                             0.0,   // muL
                             0.0);  // muR

 solve_parameters_t params;
 std::vector<std::tuple<x_index_t, double, int>> creation_ops, annihilation_ops;
 annihilation_ops = std::vector<std::tuple<x_index_t, double, int>>();
 auto pt = std::tuple<x_index_t, double, int>(0, 0.0, 0);
 creation_ops.push_back(pt);

 std::vector<dcomplex> extern_alphas;
 extern_alphas.push_back(0.);

 params.creation_ops = creation_ops;
 params.annihilation_ops = annihilation_ops;
 params.extern_alphas = extern_alphas;
 std::vector<double> measure_times = {-50., -40., -30.};
 params.measure_times = measure_times;
 std::vector<int> measure_keldysh_indices = {0, 1};
 params.measure_keldysh_indices = measure_keldysh_indices;
 std::pair<double, double> singular_thresholds = {3., 3.3};
 params.singular_thresholds = singular_thresholds;

 params.interaction_start = 150.;
 params.alpha = 0.5;
 params.U = 0.5;
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.5;
 params.w_shift = 0.0;
 params.max_perturbation_order = 6;
 params.min_perturbation_order = 0;
 params.method = 5;

 solver_core S(params);
 S.set_g0(g0.first, g0.second);
 S.run(200000, true);

 std::cout << "pn: " << S.get_pn() << std::endl;
 std::cout << "sn: " << S.get_sn() << std::endl;

 MPI::Finalize();

 return 0;
};
