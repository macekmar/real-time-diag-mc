#pragma once
#include <strings.h>

// debug only
#define CHECK_GRAY_CODE_INTEGRITY

 /// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
dcomplex recompute_sum_keldysh_indices(qmc_data_t* data, int N) {

  //int N = perturbation_order();
  if (N > 63) TRIQS_RUNTIME_ERROR << "N overflow";
  auto& matrices = data->matrices;

  //When no time is inserted, only the observable is present in the matrix
  //if (N == 0) return imag(g0_lesser(mindex(0,0,0),0));

#ifdef CHECK_GRAY_CODE_INTEGRITY
  auto mat_up = matrices[up].matrix();
  auto mat_do = matrices[down].matrix();
#endif

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
  matrices[up].regenerate();
  matrices[down].regenerate();
#endif

  dcomplex res =0 ;
  int sign = -1; // WHY IS THIS -1 ??
  auto two_to_N = uint64_t(1) << N; //shifts the bits N time to the left
  for (uint64_t n = 0; n < two_to_N; ++n) {

   // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
   // Cf Numerical Recipes, sect. 20.2
   int nlc = (n < two_to_N - 1 ? ffs(~n) : N) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                   // to 1. ~n has bites inversed compared with n.

   auto p = flip_index(matrices[up].get_x(nlc));
   matrices[up].change_one_row_and_one_col(nlc, nlc, p, p);
   matrices[down].change_one_row_and_one_col(nlc, nlc, p, p);

   res += sign * matrices[up].determinant() * matrices[down].determinant();
   sign = -sign;

   if (!std::isfinite(real(res))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
  }

  dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i !
  res = -res * i_n[(N + 1) % 4];                        // * i^(N+1)

#ifdef CHECK_GRAY_CODE_INTEGRITY
  double precision = 1.e-12;
  if (max_element(abs(mat_up - matrices[up].matrix())) > precision)
   TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_up - matrices[up].matrix()) << " Not cyclic";
  if (max_element(abs(mat_do - matrices[down].matrix())) > precision)
   TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_do - matrices[down].matrix()) << " Not cyclic";
  // check all indices of keldysh back to 0
  for (int n = 0; n < N; ++n) {
   for (auto const& mat : matrices) {
    if (mat.get_x(n).k_index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
    if (mat.get_y(n).k_index != 0) TRIQS_RUNTIME_ERROR << " Keldysh index is not 0 !!";
   }
  }
#endif

  return real(res);
 }

