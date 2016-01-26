#pragma once
#include <triqs/lattice/bravais_lattice.hpp>

std::pair<gf_latt_time_t, gf_latt_time_t> make_g0_lattice(double beta, double mu, int n_freq, double t_min, double t_max, int L,
                                                          int Lk) {

 using namespace triqs::clef;
 using namespace triqs::lattice;
 using namespace triqs::arrays;

 // construct the Brillouin zone in 2d
 auto bz = brillouin_zone{bravais_lattice{make_unit_matrix<double>(2)}};

 int n_re = 2 * (L - 1) + 1;
 int n_bz = Lk;
 int n_times = n_freq * 2 + 1;

 auto nf = [](double beta, auto eps_k) { return 1.0 / (1.0 + exp(beta * eps_k)); };
 auto mnf = [](double beta, auto eps_k) { return -(1.0 - 1.0 / (1.0 + exp(beta * eps_k))); };
 
 auto make_gxt = [&](auto func) {
  auto gxt = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>{{{Lk, Lk}, {t_min, t_max, n_times}}};
  auto gkt = gf<cartesian_product<brillouin_zone, retime>, scalar_valued, no_tail>{{{bz, n_bz}, {t_min, t_max, n_times}}};

  placeholder<0> k_;
  placeholder<1> t_;

  auto eps_k = -2 * (cos(k_(0)) + cos(k_(1))) - mu;
  gkt(k_, t_) << 1_j * exp(-1_j * eps_k * t_) * func(beta, eps_k);
  auto gx_t = curry<1>(gxt);
  auto gk_t = curry<1>(gkt);
  gx_t[t_] << inverse_fourier(gk_t[t_]);

  auto gxt_window = gf_latt_time_t{{{n_re, n_re}, {t_min, t_max, n_times}}};
  for (int i = -L; i < L; i++)
   for (int j = -L; j < L; j++) partial_eval<0>(gxt_window, mindex(i, j, 0)) = partial_eval<0>(gxt, mindex(i, j, 0));

  return gxt_window;
 };

 return {make_gxt(nf), make_gxt(mnf)};
}

