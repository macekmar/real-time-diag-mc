#include "../c++/configuration.hpp"
#include <triqs/gfs.hpp>

using namespace triqs::gfs;
using triqs::arrays::range;

#define ABS_TOL 1.e-16
#define REL_TOL 1.e-10
bool is_not_close(dcomplex a, dcomplex b) { return abs(a - b) > REL_TOL * abs(b) + ABS_TOL; }
bool is_not_close_array1(array<dcomplex, 1> a, array<dcomplex, 1> b) {
 return max_element(abs(a - b) - REL_TOL * abs(b)) > ABS_TOL;
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
 g_less_f(w_) << -0.5_j / (w_ - 1. + 1_j);
 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0.};
 g0_keldysh_alpha_t g0_alpha = g0_keldysh_alpha_t{g0, 0., alphas};

 auto a = keldysh_contour_pt{0, 1.5, 0, 0};
 auto b0 = keldysh_contour_pt{0, 0.3, 0};
 auto b1 = flip_index(b0);
 auto c0 = keldysh_contour_pt{0, -1.3, 0};
 auto c1 = flip_index(c0);
 auto d0 = keldysh_contour_pt{0, 0.65, 0};
 auto d1 = flip_index(d0);

 std::vector<keldysh_contour_pt> an_pts{
     a,
 };
 std::vector<keldysh_contour_pt> cr_pts{
     a,
 };

  auto sing_th = std::pair<double, double>{3.5, 3.5};
 //auto sing_th = std::pair<double, double>{-10000, -10000};

 Configuration config(g0_alpha, an_pts, cr_pts, 4, sing_th, true);
 dcomplex ref_weight = 1;
 auto ref_kernels = array<dcomplex, 2>(5, 2);

 // At order 0, weight is an arbitrary 1
 if (config.current_weight != 1.) return 10;
 if (config.accepted_weight != 1.) return 11;
 if (config.order != 0) return 12;

 // add a point and accept the config
 config.insert(0, b0);
 config.evaluate();
 config.accept_config();

 config.print();

 ref_kernels(0, 0) = -g0_alpha(b0, b0) * g0_alpha(b0, a);
 ref_kernels(0, 1) = g0_alpha(b1, b1) * g0_alpha(b1, a);
 ref_weight = std::abs(ref_kernels(0, 0)) + std::abs(ref_kernels(0, 1));

 if (is_not_close(config.current_weight, ref_weight)) return 20;
 if (is_not_close_array1(config.current_kernels(0, range()), ref_kernels(0, range()))) return 21;
 if (is_not_close(config.accepted_weight, ref_weight)) return 22;
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range()))) return 23;
 if (config.order != 1) return 24;

 // std::cout << ref_kernels << std::endl;
 // std::cout << config.current_kernels << std::endl;
 // std::cout << config.accepted_kernels << std::endl;

 // remove point and do not accept
 config.remove(0);
 config.evaluate();

 if (config.current_weight != 1.) return 30;
 if (is_not_close(config.accepted_weight, ref_weight)) return 31; // has not changed
 if (is_not_close_array1(config.accepted_kernels(0, range()), ref_kernels(0, range()))) return 32; // has not changed
 if (config.order != 0) return 33;

 // add two points and accept;
 config.insert2(0, 1, c0, d0);
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

 std::cout << "success" << std::endl;
 return 0;
}
