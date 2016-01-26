#pragma once
#include <strings.h>
#include <cmath>
#include <triqs/utility/time_pt.hpp>
#include <triqs/gfs.hpp>
#include "./parameters.hpp"
#include <triqs/det_manip.hpp>

using triqs::det_manip::det_manip;
using qmc_time_t = triqs::utility::time_pt;
using namespace triqs::gfs;
using gfr = gf<cartesian_product<cyclic_lattice, retime>, scalar_valued, no_tail>;

struct point {
 gf_mesh<cyclic_lattice>::index_t r;
 qmc_time_t time;   // time, in [0, t_max]. Discretized on a very thin grid. Could use a double, but simpler to compare.
 int index;         // 0 (upper contour), or 1 (lower contour)
};

bool operator==(point const &x, point const &y) {
  return (x.time == y.time) && (x.index == y.index) && (x.r == y.r);
}

/// flip the keldysh index
point flip_index(point const &t) {
 return {t.r, t.time, 1 - t.index};
}

// --------------- The QMC configuration ----------------

struct configuration { // Encode the Monte Carlo configurations

//  // quantile function of the probability for the time
//  // supposes the time scale is 1
//   qmc_time_t t_biased(qmc_time_t t_unif){ return (t_unif/(t_unif+1))*(t_max+1.);}
//  
//  // probability for the time
//  double p(qmc_time_t t){ return (t_max+1.)/(t_max*(t_max+1-t)*(double(t_max)+1-t));} 
 
 /// The function that appears in the calculation of the determinant
 // It is a lambda, but I need its explicit type below to declare matrices, so I write it explicitely.
 struct g0bar_t {
  configuration *config;

  dcomplex operator()(point const &x, point const &y) const {

   bool y_at_tmax = (y.time == config->t_max); // if true, we need a special GF for the observable

   // do not put alpha for the time_max
   if (x == y) return config->g0_lesser(mindex(0,0,0),0.0) - (y_at_tmax ? 0_j : 1_j * config->alpha);

   auto dr = x.r - y.r;
   double dtau = double(x.time)-double(y.time);
   if (x.index == 0 && y.index == 0) return (x.time > y.time ? config->g0_greater(dr, dtau) : config->g0_lesser(dr, dtau));
   if (x.index == 1 && y.index == 1) return (x.time > y.time ? config->g0_lesser(dr, dtau) : config->g0_greater(dr, dtau));
   if (x.index == 1 && y.index == 0) return config->g0_greater(dr, dtau);
   return config->g0_lesser(dr, dtau);
  }
 };

 std::vector<det_manip<g0bar_t>> matrices; ///< M-matrices for up and down
 gfr const &g0_lesser, &g0_greater;
 dcomplex alpha;
 time_segment tau_seg; ///< [0,tmax[: interval where to pick a time
 qmc_time_t t_max;     ///< time of the measures
 dcomplex sum_keldysh_indices = 1;

 // constructor
 configuration(gfr const &g0_lesser, gfr &g0_greater, double alpha, double tmax)
    : g0_lesser(g0_lesser), g0_greater(g0_greater), alpha(alpha), tau_seg(tmax), t_max(tau_seg.get_upper_pt()) {

  // Initialize the M-matrices. 100 is the initial matrix size
  for (auto spin : {up, down}) matrices.emplace_back(g0bar_t{this}, 100);

  // For up, we insert the fixed pair of times (t_max, t_max), Keldysh index +-.
  matrices[up].insert_at_end({mindex(0,0,0), t_max, 0}, {mindex(0,0,0), t_max, 1}); // C^+ C
 }

 /// Order in U (size of the up M matrix)
 int perturbation_order() const {
  assert(matrices[up].size() - 1 == matrices[down].size()); // not valid for double occ or F function
  return matrices[up].size() - 1;
 }

 //---------------------------------------------------------------
 /// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
 dcomplex recompute_sum_keldysh_indices() {
  int N = perturbation_order();
  if (N > 63) TRIQS_RUNTIME_ERROR << "N overflow";

  //When no time is inserted, only the observable is present in the matrix
  if (N == 0) return imag(g0_lesser(mindex(0,0,0),0));

  auto two_to_N = uint64_t(1) << N; //shifts he bites from N to the left
  dcomplex res =0 ;

#define CHECK_GRAY_CODE_INTEGRITY
#ifdef CHECK_GRAY_CODE_INTEGRITY
  auto det_up = matrices[up].determinant();
  auto det_do = matrices[down].determinant();
  auto mat_up = matrices[up].matrix();
  auto mat_do = matrices[down].matrix();
#endif

  if (1) { // put 1 to regenerate the matrix before each gray code
   matrices[up].regenerate();
   matrices[down].regenerate();
  }

  int sign = -1;
  for (uint64_t n = 0; n < two_to_N; ++n) {

   // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
    int nlc = (n < two_to_N - 1 ? ffs(~n) : N) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set to 1. ~n has bites inversed compared with n.

   // not true when generalizing to double occ and F.
   if (!(matrices[down].get_x(nlc) == matrices[up].get_x(nlc + 1))) TRIQS_RUNTIME_ERROR << " Internal error 118";

   auto t = flip_index(matrices[up].get_x(nlc + 1));
   matrices[up].change_one_line_and_one_col(nlc + 1, nlc + 1, t, t);
   matrices[down].change_one_line_and_one_col(nlc, nlc, t, t);
//    matrices[up].change_and_regenerate(nlc + 1, nlc + 1, t, t);
//    matrices[down].change_and_regenerate(nlc, nlc, t, t);

   res += sign * matrices[up].determinant() * matrices[down].determinant();
   sign = -sign;

   if (!std::isfinite(real(res))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
  }

   dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i !
   res = - res * i_n[(N + 1)%4]; // * i^(N+1)

#ifdef CHECK_GRAY_CODE_INTEGRITY

  double precision = 1.e-12;
  if (max_element(abs(mat_up - matrices[up].matrix())) > precision) TRIQS_RUNTIME_ERROR << make_matrix(mat_up - matrices[up].matrix())<< " Not cyclic";
  if (max_element(abs(mat_do - matrices[down].matrix())) > precision) TRIQS_RUNTIME_ERROR << make_matrix(mat_do - matrices[down].matrix())<< " Not cyclic";

  // check all indices of keldysh back to 0
  for (int n = 0; n < N; ++n) {
    if (matrices[up].get_x(n+1).index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
    if (matrices[up].get_y(n+1).index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
    if (matrices[down].get_x(n).index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
    if (matrices[down].get_y(n).index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
  }

#endif

  return real(res);
 }
};

