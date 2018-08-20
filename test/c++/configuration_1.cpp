#include "../c++/configuration.hpp"
#include <triqs/gfs.hpp>

using namespace triqs::gfs;
using triqs::arrays::range;

#define ABS_TOL 1.e-14
#define REL_TOL 1.e-10

bool is_not_close(dcomplex a, dcomplex b) {
 bool output = abs(a - b) > REL_TOL * abs(b) + ABS_TOL;
 if (output) std::cout << "Diff = " << abs(a - b) << std::endl;
 return output;
}

bool is_not_close_array1(array<dcomplex, 1> a, array<dcomplex, 1> b) {
 return max_element(abs(a - b) - REL_TOL * abs(b)) > ABS_TOL;
}

bool is_close_array2(array<dcomplex, 2> a, array<dcomplex, 2> b) {
 return max_element(abs(a - b) - REL_TOL * abs(b)) > ABS_TOL;
}

dcomplex det_2x2(g0_keldysh_alpha_t g, keldysh_contour_pt a, keldysh_contour_pt b,
                                       keldysh_contour_pt c, keldysh_contour_pt d) {
 return g(a, c) * g(b, d) - g(b, c) * g(a, d);
};

dcomplex det_3x3(g0_keldysh_alpha_t g, keldysh_contour_pt a, keldysh_contour_pt b,
                                       keldysh_contour_pt c, keldysh_contour_pt d,
                                       keldysh_contour_pt e, keldysh_contour_pt f) {
 dcomplex output = g(a, d) * (g(b, e) * g(c, f) - g(c, e) * g(b, f));
 output += -g(b, d) * (g(a, e) * g(c, f) - g(c, e) * g(a, f));
 output += g(c, d) * (g(b, e) * g(a, f) - g(a, e) * g(b, f));
 return output;
};

/**
 * Proceed to a sequence of insertions and removals of vertices, and check the
 * weight and kernel are as expected.
 * Work with a one-particle correlator.
 */
int main() {

 auto g_less = gf<retime, matrix_valued>{{-10., 10., 1001}, {2, 2}};
 auto g_less_f = make_gf_from_fourier(g_less);
 triqs::clef::placeholder<0> w_;
 g_less_f(w_) << -0.5_j / (w_ - 1.2 + 2.5_j) * array<dcomplex, 2>{{1, 1}, {1, 1}};
 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0.2};
 g0_keldysh_alpha_t g0_alpha = g0_keldysh_alpha_t{g0, 0.5, alphas};

 // external point
 auto a = keldysh_contour_pt{0, up, 1.5, 0, 0};

 // internal points
 auto b0 = keldysh_contour_pt{0, up, 0.3, 0};
 auto b1 = flip_index(b0);
 auto c0 = keldysh_contour_pt{0, up, -1.3, 0};
 auto c1 = flip_index(c0);
 auto d0 = keldysh_contour_pt{0, up, 0.65, 0};
 auto d1 = flip_index(d0);
 auto e0 = keldysh_contour_pt{0, up, -8.3, 0};
 auto e1 = flip_index(e0);

 std::vector<keldysh_contour_pt> an_pts{ a, };
 std::vector<keldysh_contour_pt> cr_pts{ a, };

 auto sing_th = std::pair<double, double>{3.5, 3.5};
 //auto sing_th = std::pair<double, double>{-10000, -10000}; // always singular

 const int max_order = 4;
 Configuration config(g0_alpha, an_pts, cr_pts, max_order, sing_th, true, false, 100);
 dcomplex ref_weight = 1;
 auto ref_kernels = array<dcomplex, 2>(max_order+1, 2); // (different points, keldysh index)

 // ---------------------------------------------------------------------------
 // At order 0, weight is an arbitrary 1
 if (config.current_weight != 1.) return 10;
 if (config.accepted_weight != 1.) return 11;
 if (config.order != 0) return 12;

 // ---------------------------------------------------------------------------
 // add a vertex and accept the config
 config.insert(vertex_t{b0.x, b0.x, b0.t, b0.k_index, 2.});
 config.evaluate();
 config.accept_config();

 config.print();

 ref_kernels(0, 0) = -g0_alpha(b0, b0) * g0_alpha(b0, a);
 ref_kernels(0, 1) = g0_alpha(b1, b1) * g0_alpha(b1, a);
 ref_kernels() *= 2.; // potential
 ref_weight = std::abs(ref_kernels(0, 0)) + std::abs(ref_kernels(0, 1));

 std::cout << ref_weight << std::endl;
 std::cout << config.current_weight << std::endl;
 std::cout << ref_kernels << std::endl;
 std::cout << config.current_kernels << std::endl;
 std::cout << config.accepted_kernels << std::endl;

 if (is_not_close(config.current_weight, ref_weight)) return 20;
 if (is_not_close_array1(config.current_kernels(0, range()), ref_kernels(0, range()))) return 21;
 if (is_not_close(config.accepted_weight, ref_weight)) return 22;
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range()))) return 23;
 if (config.order != 1) return 24;

 // ---------------------------------------------------------------------------
 // remove the vertex and do not accept
 config.remove(0);
 config.evaluate();

 if (config.current_weight != 1.) return 30;
 if (is_not_close(config.accepted_weight, ref_weight)) return 31; // has not changed
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range()))) return 32; // has not changed
 if (config.order != 0) return 33;

 // ---------------------------------------------------------------------------
 // add two vertices and accept;
 config.insert2(vertex_t{c0.x, c0.x, c0.t, c0.k_index, 1.5}, vertex_t{d0.x, d0.x, d0.t, d0.k_index, 3.2});
 config.evaluate();
 config.accept_config();

 config.print();

 ref_kernels() = 0;

 ref_kernels(0, 0) += det_2x2(g0_alpha, c0, d0, d0, a) * det_2x2(g0_alpha, c0, d0, c0, d0);
 ref_kernels(0, 0) -= det_2x2(g0_alpha, c0, d1, d1, a) * det_2x2(g0_alpha, c0, d1, c0, d1);
 ref_kernels(0, 1) -= det_2x2(g0_alpha, c1, d0, d0, a) * det_2x2(g0_alpha, c1, d0, c1, d0);
 ref_kernels(0, 1) += det_2x2(g0_alpha, c1, d1, d1, a) * det_2x2(g0_alpha, c1, d1, c1, d1);

 ref_kernels(1, 0) -= det_2x2(g0_alpha, c0, d0, c0, a) * det_2x2(g0_alpha, c0, d0, c0, d0);
 ref_kernels(1, 0) += det_2x2(g0_alpha, c1, d0, c1, a) * det_2x2(g0_alpha, c1, d0, c1, d0);
 ref_kernels(1, 1) += det_2x2(g0_alpha, c0, d1, c0, a) * det_2x2(g0_alpha, c0, d1, c0, d1);
 ref_kernels(1, 1) -= det_2x2(g0_alpha, c1, d1, c1, a) * det_2x2(g0_alpha, c1, d1, c1, d1);

 ref_kernels() *= 1.5 * 3.2; // potential

 ref_weight = sum(abs(ref_kernels));

 std::cout << ref_weight << std::endl;
 std::cout << config.current_weight << std::endl;
 std::cout << ref_kernels << std::endl;
 std::cout << config.current_kernels << std::endl;
 std::cout << config.accepted_kernels << std::endl;

 if (is_not_close(config.current_weight, ref_weight)) return 40;
 if (is_not_close_array1(config.current_kernels(0, range()), ref_kernels(0, range()))) return 41;
 if (is_not_close_array1(config.current_kernels(1, range()), ref_kernels(1, range()))) return 42;
 if (config.order != 2) return 43;

 // ---------------------------------------------------------------------------
 // add a vertex, then remove two, do not accept
 config.insert(vertex_t{e0.x, e0.x, e0.t, e0.k_index, -0.5});
 config.remove2(1, 2); // remove c and d, stays only e
 config.evaluate();

 config.print();

 // accepted kernels should not have changed
 if (is_not_close(config.accepted_weight, ref_weight)) return 50;
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range()))) return 51;
 if (is_not_close_array1(config.accepted_kernels(1, range()), ref_kernels(1, range()))) return 52;

 ref_kernels() = 0;

 ref_kernels(0, 0) = -g0_alpha(e0, e0) * g0_alpha(e0, a);
 ref_kernels(0, 1) = g0_alpha(e1, e1) * g0_alpha(e1, a);
 ref_kernels() *= -0.5; // potential
 ref_weight = std::abs(ref_kernels(0, 0)) + std::abs(ref_kernels(0, 1));

 std::cout << ref_weight << std::endl;
 std::cout << config.current_weight << std::endl;
 std::cout << ref_kernels << std::endl;
 std::cout << config.current_kernels << std::endl;
 std::cout << config.accepted_kernels << std::endl;

 if (is_not_close(config.current_weight, ref_weight)) return 53;
 if (is_not_close_array1(config.current_kernels(0, range()), ref_kernels(0, range()))) return 54;
 if (config.order != 1) return 55;

 // ---------------------------------------------------------------------------
 std::cout << "success" << std::endl;
 return 0;
}
