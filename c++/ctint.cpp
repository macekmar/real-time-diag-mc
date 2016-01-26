#include "ctint.hpp"
#include <triqs/mc_tools.hpp>
#include <triqs/det_manip.hpp>
#include <triqs/utility/time_pt.hpp>
#include "./configuration.hpp"
#include "./moves_measures.hpp"

#include <triqs/gfs.hpp>
#include <triqs/gfs/bz.hpp>
#include <triqs/gfs/m_tail.hpp>

using namespace triqs::arrays;
using namespace triqs::lattice;
//namespace h5 = triqs::h5;
using namespace triqs::gfs;
using namespace triqs::clef;
using namespace triqs::arrays;
using namespace triqs::lattice;
using triqs::utility::mindex;


// ------------ The main class of the solver ------------------------

/*
ctint_solver::ctint_solver(double beta, double Gamma, double tmax_gf0, long Nt_gf0, double epsilon_d, double muL, double muR, int GF_type)
   : cn_sn(2, 20) {

 // Prepare the non interacting GF's used to calculate the occupation or the current :
 // G0_dd_w on the dot
 // G0_dc_w_L between the dot and the left lead
 // two cases : the flat band (Werner) or the semi-circular (Xavier)

 // Construction of the empty GF's, with the correct number of points
 g0_dd_t = gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {2, 2}};
 g0_dd_w = make_gf_from_fourier(g0_dd_t);
 g0_dc_t = g0_dd_t; // gf<retime>{{-tmax_gf0, tmax_gf0, 2 * Nt_gf0 - 1}, {2, 2}};
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
   double we = w-epsilon_d;
   auto R = 0.5/(we + 1_j*Gamma);
   auto A = 0.5/(we - 1_j*Gamma);
   auto K = 1_j*Gamma/( we * we + Gamma * Gamma ) * ( nf(w - muL) + nf(w - muR) - 1.);
   return array<dcomplex, 2>{{K+R+A, K-R+A}, {K+R-A, K-R-A}};
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
   auto R = - 0.5_j * Gamma / (we + 1_j*Gamma);
   auto A =   0.5_j * Gamma / (we - 1_j*Gamma);
   auto K = 1_j * Gamma / (we * we + Gamma * Gamma)
   * ( we * (2 * nf(w - muL) - 1) + 1_j * Gamma * ( nf(w - muR) - nf(w - muL)));
    return array<dcomplex, 2> {{K+R+A, K-R+A}, {K+R-A, K-R-A}};
 };

 ///////////////////////////////////////////////////////////////////////////
 // the semi-circular band (Xavier)
 ///////////////////////////////////////////////////////////////////////////
 // Retarded self energy with semi circular sigma dos (linear chain).
 auto sigma_linear_chain = [](double omega)->dcomplex {
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
 //for one lead
 auto G0_dc_w = [&](double omega, double mu) {
  auto g = G0_dd_w(omega);                               // dot free function at omega.
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

 switch(GF_type){
   ///////////////////////////////////////////////////////////////////////////
    case 0: // case of the article, XAVIER semi circ band
      for (auto w : g0_dd_w.mesh()) g0_dd_w[w] = G0_dd_w(w);
      // we only need the left lead GF to calculate the current
      for (auto w : g0_dc_w.mesh()) g0_dc_w[w] = G0_dc_w(w, muL);
      break;
    case 1: // WERNER
      for (auto w : g0_dd_w.mesh()) g0_dd_w[w] = G0_dd_w1(w);
      // we only need the left lead GF to calculate the current
      for (auto w : g0_dc_w.mesh()) g0_dc_w[w] = G0_dc_w1(w);
      break;
 }

 // Compute the high frequency expansion, order 1/omega, and 1/omega^2
 g0_dd_w.singularity()(1) = matrix<double>{{1., 0.}, {0., -1.}};
 g0_dd_w.singularity()(2) = matrix<dcomplex>{{epsilon_d + 0_j, 1_j * Gamma}, {-1_j * Gamma, -epsilon_d + 0_j}};
 // The non interacting GF g_dc in time, obtained from the exact expression in frequency
 g0_dd_t = make_gf_from_inverse_fourier(g0_dd_w);
 g0_dc_t = make_gf_from_inverse_fourier(g0_dc_w);
}
*/



ctint_solver::ctint_solver(double beta, double mu, int n_freq, double t_min, double t_max, int L, int Lk): cn_sn(2, 20) {

  // construct the Brillouin zone in 2d
  auto bz = brillouin_zone{bravais_lattice{make_unit_matrix<double>(2)}};

  int n_re = 2*(L-1)+1;
  int n_bz = Lk;
  int n_times = n_freq * 2 + 1;

  auto nf = [](double beta, auto eps_k) { return 1.0/(1.0+exp(beta*eps_k)); };
  auto mnf = [](double beta, auto eps_k) { return -(1.0-1.0/(1.0+exp(beta*eps_k))); };
  auto make_gxt = [&](auto func) {
    auto gxt = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>{ {{Lk, Lk}, {t_min, t_max, n_times}}};
    auto gkt = gf<cartesian_product<brillouin_zone, retime>, scalar_valued, no_tail>{ {{bz, n_bz}, {t_min, t_max, n_times}}};
    
    placeholder<0> k_;
    placeholder<1> t_;

    auto eps_k = -2 * (cos(k_(0)) + cos(k_(1))) - mu;
    gkt(k_, t_) << 1_j * exp(- 1_j * eps_k * t_) * func(beta,eps_k);
    auto gx_t = curry<1>(gxt);
    auto gk_t = curry<1>(gkt);
    gx_t[t_] << inverse_fourier(gk_t[t_]);

    auto gxt_window = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>{ {{n_re,n_re}, {t_min, t_max, n_times}}};
    for (int i=-L; i<L; i++)
      for (int j=-L; j<L; j++)
        partial_eval<0>(gxt_window,mindex(i,j,0)) = partial_eval<0>(gxt,mindex(i,j,0));

    return gxt_window;

  };

  g0_lesser = make_gxt(nf);
  g0_greater = make_gxt(mnf);

}

// -------------------------------------------------------------------------
// The method that runs the qmc
void ctint_solver::solve(solve_parameters_t const &params) {

  // Construct a Monte Carlo loop
 auto qmc = triqs::mc_tools::mc_generic<dcomplex>(params.n_cycles, params.length_cycle, params.n_warmup_cycles,
                                                  params.random_name, params.random_seed, params.verbosity);
 // Prepare the configuration
 auto config = configuration{g0_lesser, g0_greater, params.alpha, params.tmax};

 // Register moves and measurements
 if(params.p_dbl<1){
  qmc.add_move(move_insert{&config, &params, qmc.rng()}, "insertion", 1.-params.p_dbl);
  qmc.add_move(move_remove{&config, &params, qmc.rng()}, "removal",   1.-params.p_dbl);
 }
 if(params.p_dbl>0){
  qmc.add_move(move_insert2{&config, &params, qmc.rng()}, "insertion2", params.p_dbl);
  qmc.add_move(move_remove2{&config, &params, qmc.rng()}, "removal2",   params.p_dbl);
 }
 qmc.add_measure(measure_cs{config, cn_sn}, "M measurement");

 // Run and collect results
 qmc.start(1.0, triqs::utility::clock_callback(params.max_time));
 qmc.collect_results(world);
}

