#include <mpi.h>
#include "../c++/solver_core.hpp"
#include "../c++/g0_flat_band.hpp"

using namespace triqs::gfs;

/// Integrated test. Test no error occurs for a typical simple use.
// Also test number of measures reported.
int main() {

 MPI::Init();

 auto g_less = gf<retime, matrix_valued>{{-100., 100., 1001}, {3, 3}};
 auto g_less_f = make_gf_from_fourier(g_less);

 // let's build an antihermitian lesser GF.
 for (auto w : g_less_f.mesh()) {
  g_less_f[w](0, 0) = -0.5_j / (w + 1.2);
  g_less_f[w](0, 1) = 1.5_j / (w - 1. + 2.5_j);
  g_less_f[w](0, 2) = 0.5_j / (w - 0.2 + 2.5_j);
  g_less_f[w](1, 0) = -conj(g_less_f[w](0, 1));
  g_less_f[w](1, 1) = -0.7_j / (w - 1.);
  g_less_f[w](1, 2) = 0.9 / (w + 0.8 + 0.5_j);
  g_less_f[w](2, 0) = -conj(g_less_f[w](0, 2));
  g_less_f[w](2, 1) = -conj(g_less_f[w](1, 2));
  g_less_f[w](2, 2) = 1.7_j / (w - 10.);
 }

 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 solve_parameters_t params;

 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 params.extern_alphas.push_back(0.);
 params.nonfixed_op = false;
 params.interaction_start = 50.;
 params.alpha = 0.5;
 params.nb_orbitals = 3;
 std::get<0>(params.potential) = {1., 0.8, 0.5, 0.5};
 std::get<1>(params.potential) = {0, 1, 1, 2};
 std::get<2>(params.potential) = {0, 1, 2, 1};

 params.U = 0.05;
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.5;
 params.w_shift = 0.5;
 params.max_perturbation_order = 3;
 params.min_perturbation_order = 0;
 params.verbosity = 1;
 params.method = 1;
 params.singular_thresholds = std::pair<double, double>{4.5, 3.3};

 solver_core S(params);
 S.set_g0(g_less, g_grea);

 S.run(20, false);
 S.run(20, true);
 std::cout << "pn: " << S.get_pn() << std::endl;

 if (S.get_nb_measures() != 20) return 1;

 S.run(20, true);
 std::cout << "pn: " << S.get_pn() << std::endl;

 if (S.get_nb_measures() != 40) return 2;

 S.collect_results(2);
 std::cout << "pn: " << S.get_pn() << std::endl;

 if (S.get_nb_measures() != 40) return 3;

 MPI::Finalize();

 return 0;
};
