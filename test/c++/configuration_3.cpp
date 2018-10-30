#include "../c++/configuration.hpp"
#include <triqs/gfs.hpp>

using namespace triqs::gfs;
using triqs::arrays::range;

bool are_equal_pts(keldysh_contour_pt const& p1, keldysh_contour_pt const& p2) {
 return p1.x == p2.x and p1.t == p2.t and p1.k_index == p2.k_index and p1.rank == p2.rank;
};

/**
 * Test that the evaluation of configurations does not change the matrices.
 */
int main() {

 MPI::Init(); // needed to create parameters (because of seed)

 // Green's function
 auto g_less = gf<retime, matrix_valued>{{-10., 10., 1001}, {1, 1}};
 auto g_less_f = make_gf_from_fourier(g_less);
 triqs::clef::placeholder<0> w_;
 g_less_f(w_) << -0.5_j / (w_ + 1_j);
 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 // Keldysh GF
 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0., 0.5};
 double alpha = 0.5;
 g0_keldysh_alpha_t g0_alpha = g0_keldysh_alpha_t{g0, alpha, alphas};

 //external points
 auto a_up = keldysh_contour_pt{0, up, 0.0, 0, 0};
 auto a_up_p = keldysh_contour_pt{0, up, 0.1, 1, 0};
 auto a_do = keldysh_contour_pt{0, down, 0.2, 0, 1};
 auto a_do_p = keldysh_contour_pt{0, down, 0.3, 0, 1};

 // internal points
 auto b = keldysh_contour_pt{0, up, 0.4, 0};
 auto c = keldysh_contour_pt{0, up, 0.5, 0};
 auto d = keldysh_contour_pt{0, up, 0.6, 0};

 const int max_order = 4;

 // parameters
 solve_parameters_t params;

 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0)); // = a_up
 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, down, 0.2, 0)); // = a_up, a_do
 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.1, 1)); // = a_up_p
 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, down, 0.3, 0)); // = a_up_p, a_do_p
 params.extern_alphas = alphas;
 params.nonfixed_op = false;
 params.interaction_start = 50.;
 params.alpha = alpha;
 params.nb_orbitals = 1;
 std::get<0>(params.potential) = {1.};
 std::get<1>(params.potential) = {0};
 std::get<2>(params.potential) = {0};

 params.U = {0.05, 0.05, 0.05};
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.5;
 params.w_shift = 0.5;
 params.max_perturbation_order = max_order;
 params.min_perturbation_order = 0;
 params.verbosity = 1;
 params.method = 1;
 params.singular_thresholds = std::pair<double, double>{3.5, 3.5};

 std::vector<keldysh_contour_pt> an_pts{a_up, a_do};
 std::vector<keldysh_contour_pt> cr_pts{a_up_p, a_do_p};

 Configuration config(g0_alpha, params);
 dcomplex ref_weight = 1;
 auto ref_kernels = array<dcomplex, 2>(max_order+1, 2); // (different points, keldysh index)

 // ---------------------------------------------------------------------------
 config.print();
 if (config.matrices[0].size() != 1) return 10;
 if (not are_equal_pts(config.matrices[0].get_x(0), a_up)) return 11;
 if (not are_equal_pts(config.matrices[0].get_y(0), a_up_p)) return 12;
 if (config.matrices[1].size() != 1) return 13;
 if (not are_equal_pts(config.matrices[1].get_x(0), a_do)) return 14;
 if (not are_equal_pts(config.matrices[1].get_y(0), a_do_p)) return 15;

 // ---------------------------------------------------------------------------
 // add a point and accept the config
 config.insert(0, vertex_t{b.x, b.x, b.t, b.k_index, 1.});
 config.evaluate();
 config.accept_config();

 config.print();

 if (config.matrices[0].size() != 2) return 20;
 if (not are_equal_pts(config.matrices[0].get_x(0), b)) return 21;
 if (not are_equal_pts(config.matrices[0].get_y(0), b)) return 22;
 if (not are_equal_pts(config.matrices[0].get_x(1), a_up)) return 23;
 if (not are_equal_pts(config.matrices[0].get_y(1), a_up_p)) return 24;
 if (config.matrices[1].size() != 2) return 25;
 if (not are_equal_pts(config.matrices[1].get_x(0), b)) return 26;
 if (not are_equal_pts(config.matrices[1].get_y(0), b)) return 27;
 if (not are_equal_pts(config.matrices[1].get_x(1), a_do)) return 28;
 if (not are_equal_pts(config.matrices[1].get_y(1), a_do_p)) return 29;

 // ---------------------------------------------------------------------------
 // remove point and do not accept
 config.remove(0);
 config.evaluate();

 config.print();

 if (config.matrices[0].size() != 1) return 30;
 if (not are_equal_pts(config.matrices[0].get_x(0), a_up)) return 31;
 if (not are_equal_pts(config.matrices[0].get_y(0), a_up_p)) return 32;
 if (config.matrices[1].size() != 1) return 33;
 if (not are_equal_pts(config.matrices[1].get_x(0), a_do)) return 34;
 if (not are_equal_pts(config.matrices[1].get_y(0), a_do_p)) return 35;

 // ---------------------------------------------------------------------------
 // add two points and accept;
 config.insert2(0, 1, vertex_t{c.x, c.x, c.t, c.k_index, 1.}, vertex_t{d.x, d.x, d.t, d.k_index, 1.});
 config.evaluate();
 config.accept_config();

 config.print();

 if (config.matrices[0].size() != 3) return 40;
 if (not are_equal_pts(config.matrices[0].get_x(0), c)) return 41;
 if (not are_equal_pts(config.matrices[0].get_y(0), c)) return 42;
 if (not are_equal_pts(config.matrices[0].get_x(1), d)) return 43;
 if (not are_equal_pts(config.matrices[0].get_y(1), d)) return 44;
 if (not are_equal_pts(config.matrices[0].get_x(2), a_up)) return 45;
 if (not are_equal_pts(config.matrices[0].get_y(2), a_up_p)) return 46;

 if (config.matrices[1].size() != 3) return 50;
 if (not are_equal_pts(config.matrices[1].get_x(0), c)) return 51;
 if (not are_equal_pts(config.matrices[1].get_y(0), c)) return 52;
 if (not are_equal_pts(config.matrices[1].get_x(1), d)) return 53;
 if (not are_equal_pts(config.matrices[1].get_y(1), d)) return 54;
 if (not are_equal_pts(config.matrices[1].get_x(2), a_do)) return 55;
 if (not are_equal_pts(config.matrices[1].get_y(2), a_do_p)) return 56;

 std::cout << "nb cofactors = " << config.nb_cofact << std::endl;
 std::cout << "nb inverse = " << config.nb_inverse << std::endl;
 std::cout << "success" << std::endl;
 MPI::Finalize();
 return 0;
}
