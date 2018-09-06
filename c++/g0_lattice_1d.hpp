#pragma once
#include <triqs/gfs.hpp>
#include <math.h>

using namespace triqs::gfs;
using dcomplex = std::complex<double>;

template<typename ReturnT, typename FuncT>
ReturnT trapz(FuncT f, double lower, double upper, int Nb_pts) {
 auto res = 0.5 * (f(lower) + f(upper));
 auto dx = (upper - lower) / (Nb_pts - 1);
 auto x_m = upper - 0.5*dx;
 for (auto x = lower + dx; x < x_m; x += dx) {res += f(x);}
 return res * dx;
};


/**
 * Compute the Green's functions of a 1D periodic lattice with nearest-neighboors coupling.
 * `beta` is the inverse temperature. If negative, temperature is zero.
 *
 * TODO: complete description
 */
std::pair<gf_view<retime>, gf_view<retime>> make_g0_lattice_1d(double beta, double mu, double epsilon, double hop, double tmax_gf0, int Nt_gf0, int nb_sites, int Nb_k_pts) {
 double pi = M_PI; // from math.h
 if (std::abs(hop) < 1e-10)
  TRIQS_RUNTIME_ERROR << "Atomic limit is not supported."; // for T=0 in particular

 // Construction of the empty GF's, with the correct number of points
 auto g0_greater = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {nb_sites, nb_sites}};
 auto g0_lesser = g0_greater;

 // energy spectrum
 auto spectrum = [&](double k) -> double { return epsilon - 2 * hop * std::cos(k); };

 // fermi distribution
 auto fermi = [&](double e) -> double { return 1 / (1 + std::exp(beta*(e - mu))); };

 dcomplex less_value, grea_value;
 double bound, lo_bound, up_bound;
 int j;

 for (auto t : g0_greater.mesh()) {
  for (int delta_site = 1 - nb_sites; delta_site < nb_sites; ++delta_site) {
   // we use the symmetry of spectrum(k) to reduce the integration range

   // lesser
   if (beta >= 0) {
    less_value = 1_j / pi * trapz<dcomplex>([&](double k) -> dcomplex {return fermi(spectrum(k)) * std::exp(-1_j*t*spectrum(k)) * std::cos(k*delta_site);}, 0, pi, Nb_k_pts);
   } else { // T=0, fermi is a heaviside
    bound = 0.5 * (epsilon - mu) / hop;
    bound = std::max(-1., std::min(1., bound)); // restrict to [-1, 1]
    bound = std::acos(bound);
    if (hop < 0) {lo_bound = bound; up_bound = pi;}
    else {lo_bound = 0; up_bound = bound;}
    less_value = 1_j / pi * trapz<dcomplex>([&](double k) -> dcomplex {return std::exp(-1_j*t*spectrum(k)) * std::cos(k*delta_site);}, lo_bound, up_bound, Nb_k_pts);
   }

   // greater
   grea_value = less_value - 1_j / pi * trapz<dcomplex>([&](double k) -> dcomplex {return std::exp(-1_j*t*spectrum(k)) * std::cos(k*delta_site);}, 0, pi, Nb_k_pts);

   // fill gfs
   for (int i = std::max(0, -delta_site); i < nb_sites + std::min(0, -delta_site); ++i) {
    j = i + delta_site;
    g0_lesser[t](i, j) = less_value;
    g0_greater[t](i, j) = grea_value;
   };

  };
 };

 return {g0_lesser, g0_greater};

};
