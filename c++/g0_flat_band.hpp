#pragma once

std::pair<gf_latt_time_t, gf_latt_time_t> make_g0_flat_band(double beta, double Gamma, double tmax_gf0, long Nt_gf0,
                                                            double epsilon_d, double muL, double muR) {
 // Prepare the non interacting GF's used to calculate the occupation or the current :
 // G0_dd_w on the dot
 // G0_dc_w_L between the dot and the left lead
 // two cases : the flat band (Werner) or the semi-circular (Xavier)

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
 // the flat band (Werner)
 ///////////////////////////////////////////////////////////////////////////
 // The non interacting dot GF's in frequency (2*2 matrix with Keldysh indices)
 // From Werner paper

 // Give the same results as the following function (used for verifications)
 //  auto G0_dd_w1 = [&](double w) {
 //    dcomplex fact = Gamma * 1_j / ((w - epsilon_d) * (w - epsilon_d) + Gamma * Gamma);
 //    dcomplex temp2 = (nf(w - muL) + nf(w - muR)) * fact;
 //    auto gdc00 = 1.0 / (w - epsilon_d + Gamma * 1_j) + temp2;
 //    auto gdc10 = temp2 - 2.0 * fact;
 //    auto gdc11 = -1.0 / (w - epsilon_d - Gamma * 1_j) + temp2;
 //    return array<dcomplex, 2>{{gdc00, temp2},{gdc10, gdc11}};
 //  };

 auto G0_dd_w = [&](double w) {
  double we = w - epsilon_d;
  auto R = 0.5 / (we + 1_j * Gamma);
  auto A = 0.5 / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) * (nf(w - muL) + nf(w - muR) - 1.);
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };

 //////////////////////////////////////////////////////////////////////
 //  for one lead

 //  //Give the same results as the following function (used for verifications)
 //  auto G0_dc_w1 = [&](double w) {
 //    auto g = G0_dd_w1(w);
 //    // we take Gamma_L=Gamma_R=Gamma/2
 //    double nl = nf(w - muL);
 //
 //    auto SR = -0.5_j*Gamma;
 //    auto SA =  0.5_j*Gamma;
 //    auto SK =  0.5_j*Gamma * (4*nl-2);
 //    auto sigma_00 = ( SR+SA+SK)/2;
 //    auto sigma_01 = ( SR-SA-SK)/2;
 //    auto sigma_10 = (-SR+SA-SK)/2;
 //    auto sigma_11 = (-SR-SA+SK)/2;
 //
 //    auto gdc00 = g(0, 0) * sigma_00 + g(0, 1) * sigma_10;
 //    auto gdc01 = -g(0, 0) * sigma_01 - g(0, 1) * sigma_11;
 //    auto gdc10 = g(1, 0) * sigma_00 + g(1, 1) * sigma_10;
 //    auto gdc11 = -g(1, 0) * sigma_01 - g(1, 1) * sigma_11;
 //    return array<dcomplex, 2>{{gdc00, gdc01}, {gdc10, gdc11}};
 //  };

 auto G0_dc_w = [&](double w) {
  double we = w - epsilon_d;
  auto R = -0.5_j * Gamma / (we + 1_j * Gamma);
  auto A = 0.5_j * Gamma / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) * (we * (2 * nf(w - muL) - 1) + 1_j * Gamma * (nf(w - muR) - nf(w - muL)));
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };

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

 return {g0_lesser_t, g0_greater_t};
}
