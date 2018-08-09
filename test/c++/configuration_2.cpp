#include "../c++/configuration.hpp"
#include <triqs/gfs.hpp>

using namespace triqs::gfs;
using triqs::arrays::range;

#define ABS_TOL 1.e-15
#define REL_TOL 1.e-10
bool is_not_close(dcomplex a, dcomplex b) {
 bool output = abs(a - b) > REL_TOL * abs(b) + ABS_TOL;
 if (output) std::cout << "Diff = " << abs(a - b) << std::endl;
 return output;
}
bool is_not_close_array1(array<dcomplex, 1> a, array<dcomplex, 1> b) {
 bool output = max_element(abs(a - b) - REL_TOL * abs(b)) > ABS_TOL;
 if (output) std::cout << "Diff = " << max_element(abs(a - b)) << std::endl;
 return output;
}
bool is_close_array2(array<dcomplex, 2> a, array<dcomplex, 2> b) {
 return max_element(abs(a - b) - REL_TOL * abs(b)) > ABS_TOL;
}

dcomplex det_2x2(g0_keldysh_alpha_t g, keldysh_contour_pt a, keldysh_contour_pt b, keldysh_contour_pt c,
                 keldysh_contour_pt d) {
 return g(a, c) * g(b, d) - g(b, c) * g(a, d);
};

dcomplex det_3x3(g0_keldysh_alpha_t g, keldysh_contour_pt a, keldysh_contour_pt b, keldysh_contour_pt c,
                 keldysh_contour_pt d, keldysh_contour_pt e, keldysh_contour_pt f) {
 dcomplex output = g(a, d) * (g(b, e) * g(c, f) - g(c, e) * g(b, f));
 output += -g(b, d) * (g(a, e) * g(c, f) - g(c, e) * g(a, f));
 output += g(c, d) * (g(b, e) * g(a, f) - g(a, e) * g(b, f));
 return output;
};

int main() {

 auto g_less = gf<retime, matrix_valued>{{-10., 10., 1001}, {2, 2}};
 auto g_less_f = make_gf_from_fourier(g_less);
 triqs::clef::placeholder<0> w_;
 g_less_f(w_) << -0.5_j / (w_ - 1.2 + 2.5_j);
 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0., 1.5};
 g0_keldysh_alpha_t g0_alpha = g0_keldysh_alpha_t{g0, 0.5, alphas};

 auto a_up = keldysh_contour_pt{0, up, 1.5, 0, 0};
 auto a_up_p = keldysh_contour_pt{0, up, 1.4, 1, 0};
 auto a_do = keldysh_contour_pt{0, down, 0.5, 0, 1};
 auto b0 = keldysh_contour_pt{0, up, 0.3, 0};
 auto b1 = flip_index(b0);
 auto c0 = keldysh_contour_pt{0, up, -1.3, 0};
 auto c1 = flip_index(c0);
 auto d0 = keldysh_contour_pt{0, up, 0.65, 0};
 auto d1 = flip_index(d0);

 std::vector<keldysh_contour_pt> an_pts{a_up, a_do};
 std::vector<keldysh_contour_pt> cr_pts{a_up_p, a_do};

 auto sing_th = std::pair<double, double>{3.5, 3.5};
 // auto sing_th = std::pair<double, double>{-10000, -10000};

 Configuration config(g0_alpha, an_pts, cr_pts, 4, sing_th, true, false, 100);
 dcomplex ref_weight = 1;
 auto ref_kernels = array<dcomplex, 2>(5, 2);

 // At order 0, weight is an arbitrary 1
 if (config.current_weight != 1.) return 10;
 if (config.accepted_weight != 1.) return 11;
 if (config.order != 0) return 12;

 // add a point and accept the config
 config.insert(0, vertex_t{b0.x, b0.x, b0.t, b0.k_index});
 config.evaluate();
 config.accept_config();

 config.print();

 ref_kernels(0, 0) = -g0_alpha(b0, a_up_p) * det_2x2(g0_alpha, b0, a_do, b0, a_do);
 ref_kernels(0, 1) = +g0_alpha(b1, a_up_p) * det_2x2(g0_alpha, b1, a_do, b1, a_do);
 ref_kernels(1, 1) = +g0_alpha(b0, b0) * det_2x2(g0_alpha, b0, a_do, b0, a_do); // only index of a_up_p (1)
 ref_kernels(1, 1) += -g0_alpha(b1, b1) * det_2x2(g0_alpha, b1, a_do, b1, a_do);
 ref_weight = sum(abs(ref_kernels(0, range())));
 ref_weight += sum(abs(ref_kernels(1, range())));

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

 // remove point and do not accept
 config.remove(0);
 config.evaluate();

 if (config.current_weight != 1.) return 30;
 if (is_not_close(config.accepted_weight, ref_weight)) return 31; // has not changed
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range())))
  return 32; // has not changed
 if (config.order != 0) return 33;

 // add two points and accept;
 config.insert2(0, 1, vertex_t{c0.x, c0.x, c0.t, c0.k_index}, vertex_t{d0.x, d0.x, d0.t, d0.k_index});
 config.evaluate();
 config.accept_config();

 config.print();

 ref_kernels() = 0;

 ref_kernels(0, 0) += det_2x2(g0_alpha, c0, d0, d0, a_up_p) * det_3x3(g0_alpha, c0, d0, a_do, c0, d0, a_do);
 ref_kernels(0, 0) -= det_2x2(g0_alpha, c0, d1, d1, a_up_p) * det_3x3(g0_alpha, c0, d1, a_do, c0, d1, a_do);
 ref_kernels(0, 1) -= det_2x2(g0_alpha, c1, d0, d0, a_up_p) * det_3x3(g0_alpha, c1, d0, a_do, c1, d0, a_do);
 ref_kernels(0, 1) += det_2x2(g0_alpha, c1, d1, d1, a_up_p) * det_3x3(g0_alpha, c1, d1, a_do, c1, d1, a_do);

 ref_kernels(1, 0) -= det_2x2(g0_alpha, c0, d0, c0, a_up_p) * det_3x3(g0_alpha, c0, d0, a_do, c0, d0, a_do);
 ref_kernels(1, 0) += det_2x2(g0_alpha, c1, d0, c1, a_up_p) * det_3x3(g0_alpha, c1, d0, a_do, c1, d0, a_do);
 ref_kernels(1, 1) += det_2x2(g0_alpha, c0, d1, c0, a_up_p) * det_3x3(g0_alpha, c0, d1, a_do, c0, d1, a_do);
 ref_kernels(1, 1) -= det_2x2(g0_alpha, c1, d1, c1, a_up_p) * det_3x3(g0_alpha, c1, d1, a_do, c1, d1, a_do);

 // only index of a_up_p (1)
 ref_kernels(2, 1) += det_2x2(g0_alpha, c0, d0, c0, d0) * det_3x3(g0_alpha, c0, d0, a_do, c0, d0, a_do);
 ref_kernels(2, 1) -= det_2x2(g0_alpha, c1, d0, c1, d0) * det_3x3(g0_alpha, c1, d0, a_do, c1, d0, a_do);
 ref_kernels(2, 1) -= det_2x2(g0_alpha, c0, d1, c0, d1) * det_3x3(g0_alpha, c0, d1, a_do, c0, d1, a_do);
 ref_kernels(2, 1) += det_2x2(g0_alpha, c1, d1, c1, d1) * det_3x3(g0_alpha, c1, d1, a_do, c1, d1, a_do);

 ref_weight = sum(abs(ref_kernels));

 std::cout << ref_weight << std::endl;
 std::cout << config.current_weight << std::endl;
 std::cout << ref_kernels << std::endl;
 std::cout << config.current_kernels << std::endl;
 std::cout << config.accepted_kernels << std::endl;

 if (is_not_close(config.current_weight, ref_weight)) return 40;
 if (is_not_close_array1(config.current_kernels(0, range()), ref_kernels(0, range()))) return 41;
 if (is_not_close_array1(config.current_kernels(1, range()), ref_kernels(1, range()))) return 42;
 if (is_not_close_array1(config.current_kernels(2, range()), ref_kernels(2, range()))) return 43;
 if (config.order != 2) return 44;

 std::cout << "success" << std::endl;
 return 0;
}
