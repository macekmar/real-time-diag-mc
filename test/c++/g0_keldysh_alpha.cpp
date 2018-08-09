#include <triqs/gfs.hpp>
#include "../c++/qmc_data.hpp"

using namespace triqs::gfs;

int main() {
 auto g_less = gf<retime, matrix_valued>{{-10., 10., 1001}, {2, 2}};
 auto g_less_f = make_gf_from_fourier(g_less);
 triqs::clef::placeholder<0> w_;
 g_less_f(w_) << -0.5_j / (w_ + 1_j);
 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0.1, 1.};
 auto g0_alpha = g0_keldysh_alpha_t{g0, 0.5, alphas};

 keldysh_contour_pt a, b;

 // matching internal points
 a = keldysh_contour_pt{0, up, 1.5, 0, -1};
 b = keldysh_contour_pt{0, up, 1.5, 0, -1};
 if (g0_alpha(a, b) != g0(a, b) - 0.5_j) return 1;

 // mismatching (in time) internal points
 a = keldysh_contour_pt{0, up, 1.5, 0, -1};
 b = keldysh_contour_pt{0, up, 2.0, 0, -1};
 if (g0_alpha(a, b) != g0(a, b)) return 2;

 // mismatching (in space/orbital) internal points
 a = keldysh_contour_pt{0, up, 1.5, 0, -1};
 b = keldysh_contour_pt{1, up, 1.5, 0, -1};
 if (g0_alpha(a, b) != g0(a, b)) return 3;

 // internal and external points, otherwise equal
 a = keldysh_contour_pt{0, up, 1.5, 0, -1};
 b = keldysh_contour_pt{0, up, 1.5, 0, 1};
 if (g0_alpha(a, b) != g0(a, b)) return 4;

 // external and internal points, otherwise equal
 a = keldysh_contour_pt{0, up, 1.5, 0, 1};
 b = keldysh_contour_pt{0, up, 1.5, 0, -1};
 if (g0_alpha(a, b) != g0(a, b)) return 5;

 // equal but mismatching external points
 a = keldysh_contour_pt{0, up, 1.5, 0, 1};
 b = keldysh_contour_pt{0, up, 1.5, 0, 0};
 if (g0_alpha(a, b) != g0(a, b)) return 6;

 // equal and matching external points
 a = keldysh_contour_pt{0, up, 1.5, 0, 0};
 b = keldysh_contour_pt{0, up, 1.5, 0, 0};
 if (g0_alpha(a, b) != g0(a, b) - 0.1_j) return 7;

 // equal and matching external points
 a = keldysh_contour_pt{0, up, 1.5, 0, 1};
 b = keldysh_contour_pt{0, up, 1.5, 0, 1};
 if (g0_alpha(a, b) != g0(a, b) - 1_j) return 8;

 // different (in time) but matching external points
 a = keldysh_contour_pt{0, up, 0.5, 0, 1};
 b = keldysh_contour_pt{0, up, 1.5, 0, 1};
 if (g0_alpha(a, b) != g0(a, b) - 1_j) return 9;

 // different (in space/orbital) but matching external points
 a = keldysh_contour_pt{2, up, 0.5, 0, 1};
 b = keldysh_contour_pt{0, up, 0.5, 0, 1};
 if (g0_alpha(a, b) != g0(a, b) - 1_j) return 10;

 // different (in keldsh index) but matching external points
 a = keldysh_contour_pt{2, up, 0.5, 1, 1};
 b = keldysh_contour_pt{0, up, 0.5, 0, 1};
 if (g0_alpha(a, b) != g0(a, b) - 1_j) return 11;

 // matching internal points with different keldysh indices
 a = keldysh_contour_pt{1, up, 0.5, 0, -1};
 b = keldysh_contour_pt{1, up, 0.5, 1, -1};
 if (g0_alpha(a, b) != g0(a, b) - 0.5_j) return 12;

 return 0;
}

