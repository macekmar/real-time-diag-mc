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

 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 params.extern_alphas.push_back(0.);
 params.nonfixed_op = false;
 params.interaction_start = 150.;
 params.alpha = 0.5;
 params.nb_orbitals = 1;
 std::get<0>(params.potential).push_back(1.);
 std::get<1>(params.potential).push_back(0);
 std::get<2>(params.potential).push_back(0);

 params.U = 0.5;
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.5;
 params.w_shift = 0.0;
 params.max_perturbation_order = 6;
 params.min_perturbation_order = 0;
 params.verbosity = 1;
 params.method = 1;
 params.nb_bins = 100;
 params.singular_thresholds = std::pair<double, double>{4.5, 3.3};

 solver_core S(params);
 S.set_g0(g0.first, g0.second);

 S.run(200, true);
 std::cout << "pn: " << S.get_pn() << std::endl;

 S.run(200, true);
 std::cout << "pn: " << S.get_pn() << std::endl;
 std::cout << "kernels:" << std::endl << S.get_kernels() << std::endl;

 S.collect_results(2);
 std::cout << "pn: " << S.get_pn() << std::endl;
 std::cout << "kernels:" << std::endl << S.get_kernels() << std::endl;

 MPI::Finalize();

 return 0;
};
