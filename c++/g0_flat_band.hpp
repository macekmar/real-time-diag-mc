#pragma once
#include <triqs/gfs.hpp>
#include <boost/math/special_functions/expint.hpp>

using namespace triqs::gfs;

// The flat band (Werner)
gf_view<refreq> make_g0c_flat_band_freq(double beta, double Gamma, double tmax_gf0, int Nt_gf0,
                                        double epsilon_d, double muL, double muR) {
 // Prepare the non interacting GF's on the dot only :
 // G0_dd_w on the dot

 // Construction of the empty GF's, with the correct number of points
 auto g0_keldysh_t = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {2, 2}};
 auto g0_keldysh_w = make_gf_from_fourier(g0_keldysh_t);

 // Fermi function
 auto nf = [&](double omega) {
  return beta > 0 ? 1. / (1. + std::exp(beta * omega))
                  : (beta < 0. ? ((omega < 0. ? 1. : (omega > 0. ? 0. : 0.5))) : 0.5);
 };

 // The non interacting dot GF's in frequency (2*2 matrix with Keldysh indices)
 auto G0_dd_w = [&](double w) {
  double we = w - epsilon_d;
  auto R = 0.5 / (we + 1_j * Gamma);
  auto A = 0.5 / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) * (nf(w - muL) + nf(w - muR) - 1.);
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };

 for (auto w : g0_keldysh_w.mesh()) {
  g0_keldysh_w[w]() = G0_dd_w(w);
 }

 return g0_keldysh_w;
}

// The flat band (Werner)
std::pair<gf_view<refreq>, gf_view<refreq>> make_g0_flat_band_freq(double beta, double Gamma, double tmax_gf0,
                                                                   int Nt_gf0, double epsilon_d, double muL,
                                                                   double muR) {
 // Prepare the non interacting GF's used to calculate the occupation or the current :
 // G0_dd_w on the dot
 // G0_dc_w between the dot and the left lead

 // Construction of the empty GF's, with the correct number of points
 auto g0_greater_t = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {2, 2}};
 auto g0_greater_w = make_gf_from_fourier(g0_greater_t);
 auto g0_lesser_t = g0_greater_t;
 auto g0_lesser_w = g0_greater_w;

 // Fermi function
 auto nf = [&](double omega) {
  return beta > 0 ? 1. / (1. + std::exp(beta * omega))
                  : (beta < 0. ? ((omega < 0. ? 1. : (omega > 0. ? 0. : 0.5))) : 0.5);
 };

 // The non interacting dot GF's in frequency (2*2 matrix with Keldysh indices)
 auto G0_dd_w = [&](double w) {
  double we = w - epsilon_d;
  auto R = 0.5 / (we + 1_j * Gamma);
  auto A = 0.5 / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) * (nf(w - muL) + nf(w - muR) - 1.);
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };

 // For one lead
 auto G0_dc_w = [&](double w) {
  double we = w - epsilon_d;
  auto R = -0.5_j * Gamma / (we + 1_j * Gamma);
  auto A = 0.5_j * Gamma / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) *
           (we * (2 * nf(w - muL) - 1) + 1_j * Gamma * (nf(w - muR) - nf(w - muL)));
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };

 for (auto w : g0_greater_w.mesh()) {
  auto g0_dd = G0_dd_w(w);
  g0_lesser_w[w](0, 0) = g0_dd(0, 1);
  g0_greater_w[w](0, 0) = g0_dd(1, 0);
  auto g0_dc = G0_dc_w(w);
  g0_lesser_w[w](0, 1) = g0_dc(0, 1);
  g0_greater_w[w](0, 1) = g0_dc(1, 0);
  // FIXME set lower components to zero
  g0_greater_w[w](1, 0) = 0.0;
  g0_greater_w[w](1, 1) = 0.0;
  g0_lesser_w[w](1, 0) = 0.0;
  g0_lesser_w[w](1, 1) = 0.0;
 }

 return {g0_lesser_w, g0_greater_w};
}

std::pair<gf_view<retime>, gf_view<retime>> make_g0_flat_band(double beta, double Gamma, double tmax_gf0,
                                                              int Nt_gf0, double epsilon_d, double muL,
                                                              double muR) {
//  auto g0_w = make_g0_flat_band_freq(beta, Gamma, tmax_gf0, Nt_gf0, epsilon_d, muL, muR);

//  // The non interacting GF in time, obtained from the exact expression in frequency
//  auto g0_lesser_t = make_gf_from_inverse_fourier(g0_w.first);
//  auto g0_greater_t = make_gf_from_inverse_fourier(g0_w.second);

//  return {g0_lesser_t, g0_greater_t};

    auto const time_mesh = gf_mesh<retime>({-tmax_gf0, tmax_gf0, Nt_gf0});
    auto g0_lesser_up = gf<retime>{time_mesh, {2, 2}};
    auto g0_greater_up = gf<retime>{time_mesh, {2, 2}}; 

    auto const g0_lesser_values = [Gamma](double time) -> dcomplex {
        using namespace boost::math;
        using namespace boost::math::double_constants;

        if (time == 0.0) {
        return 0.5_j;
        }
        auto Gt = Gamma * time;
        auto real_part = (std::exp(Gt) * expint(-Gt) - std::exp(-Gt) * expint(Gt)) / (2 * pi);
        return real_part + 0.5_j * std::exp(-std::abs(Gt));
    };

    dcomplex val = 0.;
    for (auto t : time_mesh) {
        val = g0_lesser_values(t);
        g0_lesser_up[t](0, 0) = val;
        g0_greater_up[t](0, 0) = std::conj(val);
        g0_lesser_up[t](0, 1) = 0.0;
        g0_lesser_up[t](1, 0) = 0.0;
        g0_lesser_up[t](1, 1) = 0.0;
        g0_greater_up[t](0, 1) = 0.0;
        g0_greater_up[t](1, 0) = 0.0;
        g0_greater_up[t](1, 1) = 0.0;
    }
return {g0_lesser_up, g0_greater_up};
}
