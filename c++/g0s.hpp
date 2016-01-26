#pragma once

std::pair<gf_latt_time_t, gf_latt_time_t> make_g0_flat_band(double beta, double Gamma, double tmax_gf0, long Nt_gf0,
                                                            double epsilon_d, double muL, double muR, int GF_type) {
 // Prepare the non interacting GF's used to calculate the occupation or the current :
 // G0_dd_w on the dot
 // G0_dc_w_L between the dot and the left lead
 // two cases : the flat band (Werner) or the semi-circular (Xavier)

 // Construction of the empty GF's, with the correct number of points
 g0_dd_t = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {2, 2}};
 g0_dd_w = make_gf_from_fourier(g0_dd_t);
 g0_dc_t = g0_dd_t; 
 g0_dc_w = g0_dd_w;

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

 auto G0_dd_w1 = [&](double w) {
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

 auto G0_dc_w1 = [&](double w) {
  double we = w - epsilon_d;
  auto R = -0.5_j * Gamma / (we + 1_j * Gamma);
  auto A = 0.5_j * Gamma / (we - 1_j * Gamma);
  auto K = 1_j * Gamma / (we * we + Gamma * Gamma) * (we * (2 * nf(w - muL) - 1) + 1_j * Gamma * (nf(w - muR) - nf(w - muL)));
  return array<dcomplex, 2>{{K + R + A, K - R + A}, {K + R - A, K - R - A}};
 };



 for (auto w : g0_dd_w.mesh()) g0_dd_w[w] = G0_dd_w1(w);
 
 // we only need the left lead GF to calculate the current
 for (auto w : g0_dc_w.mesh()) g0_dc_w[w] = G0_dc_w1(w);

 // Compute the high frequency expansion, order 1/omega, and 1/omega^2
 g0_dd_w.singularity()(1) = matrix<double>{{1., 0.}, {0., -1.}};
 g0_dd_w.singularity()(2) = matrix<dcomplex>{{epsilon_d + 0_j, 1_j * Gamma}, {-1_j * Gamma, -epsilon_d + 0_j}};
 // The non interacting GF g_dc in time, obtained from the exact expression in frequency
 g0_dd_t = make_gf_from_inverse_fourier(g0_dd_w);
 g0_dc_t = make_gf_from_inverse_fourier(g0_dc_w);

}

// ------------------------------------------------------------------------------------------

std::pair<gf_latt_time_t, gf_latt_time_t> make_g0_semi_circular(double beta, double Gamma, double tmax_gf0, long Nt_gf0,
                                                                double epsilon_d, double muL, double muR, int GF_type) {


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

      for (auto w : g0_dd_w.mesh()) g0_dd_w[w] = G0_dd_w(w);
      // we only need the left lead GF to calculate the current
      for (auto w : g0_dc_w.mesh()) g0_dc_w[w] = G0_dc_w(w, muL);

 // Compute the high frequency expansion, order 1/omega, and 1/omega^2
 g0_dd_w.singularity()(1) = matrix<double>{{1., 0.}, {0., -1.}};
 g0_dd_w.singularity()(2) = matrix<dcomplex>{{epsilon_d + 0_j, 1_j * Gamma}, {-1_j * Gamma, -epsilon_d + 0_j}};
 // The non interacting GF g_dc in time, obtained from the exact expression in frequency
 g0_dd_t = make_gf_from_inverse_fourier(g0_dd_w);
 g0_dc_t = make_gf_from_inverse_fourier(g0_dc_w);
}



}



