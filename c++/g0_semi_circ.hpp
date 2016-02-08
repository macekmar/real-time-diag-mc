#pragma once
#include <triqs/gfs.hpp>

std::pair<gf<retime>, gf<retime>> make_g0_semi_circular(double beta, double Gamma, double tmax_gf0, long Nt_gf0, double epsilon_d,
                                                        double muL, double muR) {

 // Construction of the empty GF's, with the correct number of points
 g0_greater_t = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {1, 2}};
 g0_greater_w = make_gf_from_fourier(g0_greater_t);
 g0_lesser_t = g0_greater_t;
 g0_lesser_w = g0_greater_w;

 // Fermi function
 auto nf = [&](double omega) {
  return beta > 0 ? 1 / (1 + std::exp(beta * omega)) : (beta < 0 ? ((omega < 0 ? 1 : (omega > 0 ? 0 : 0.5))) : 0.5);
 };

 ///////////////////////////////////////////////////////////////////////////
 // the semi-circular band (Xavier)
 ///////////////////////////////////////////////////////////////////////////
 // Retarded self energy with semi circular sigma dos (linear chain).
 auto sigma_linear_chain = [](double omega) -> dcomplex {
  omega = omega / 2;
  if (std::abs(omega) < 1) return dcomplex{omega, -std::sqrt(1 - omega * omega)};
  if (omega > 1) return omega - std::sqrt(omega * omega - 1);
  return omega + std::sqrt(omega * omega - 1);
 };

 // The non interacting dot GF's in frequency (2*2 matrix with Keldysh indices)
 // From XW computation
 auto G0_dd_w = [&](double omega) {
  dcomplex gr = 1 / (omega - epsilon_d - 2 * Gamma * sigma_linear_chain(omega));
  dcomplex fac = 2_j * gr * conj(gr);
  dcomplex gam_gg = -1 * Gamma * imag(sigma_linear_chain(omega)) * fac;
  dcomplex temp = (nf(omega - muL) + nf(omega - muR)) * gam_gg;
  dcomplex temp2 = temp - 2 * gam_gg;
  return array<dcomplex, 2>{{gr + temp, temp}, {temp2, -conj(gr) + temp}};
 };

 ////////////////////////////////////////////////////////////////////////////
 // for one lead
 auto G0_dc_w = [&](double omega, double mu) {
  auto g = G0_dd_w(omega);                          // dot free function at omega.
  auto delta_r = Gamma * sigma_linear_chain(omega); // both chains are the same
  auto delta_01 = -2_j * imag(delta_r) * nf(omega - mu);
  auto delta_10 = 2_j * imag(delta_r) * (1 - nf(omega - mu));
  auto delta_00 = delta_r + delta_01;
  auto delta_11 = delta_10 - delta_r;
  auto gdc00 = (g(0, 0) * delta_00 - g(0, 1) * delta_10);
  auto gdc01 = (g(0, 0) * delta_01 - g(0, 1) * delta_11);
  auto gdc10 = (g(1, 0) * delta_00 - g(1, 1) * delta_10);
  auto gdc11 = (g(1, 0) * delta_01 - g(1, 1) * delta_11);
  return array<dcomplex, 2>{{gdc00, gdc01}, {gdc10, gdc11}};
 };
 ///////////////////////////////////////////////////////////////////////////

 for (auto w : g0_dd_w.mesh()) {
  auto g0_dd = G0_dd_w(w);
  g0_greater_w[w](0, 0) = g0_dd(0, 1);
  g0_lesser_w[w](0, 0) = g0_dd(1, 0);
  auto g0_dc = G0_dc_w(w, muL); // we only need the left lead GF to calculate the current
  g0_greater_w[w](0, 1) = g0_dc(0, 1);
  g0_lesser_w[w](0, 1) = g0_dc(1, 0);
 }
 // The non interacting GF in time, obtained from the exact expression in frequency
 g0_greater_t = make_gf_from_inverse_fourier(g0_greater_w);
 g0_lesser_t = make_gf_from_inverse_fourier(g0_lesser_w);

 return { g0_lesser_t, g0_greater_t };
}

