#include "./qmc_data.hpp"

//#define CHECK_GRAY_CODE_INTEGRITY
#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

/// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t>& matrix_up, det_manip<g0_keldysh_t>& matrix_down, int k) {

 if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";

 if (k == 0) {
  return matrix_up.determinant() * matrix_down.determinant();
 }

#ifdef CHECK_GRAY_CODE_INTEGRITY
 auto mat_up = matrix_up.matrix();
 auto mat_do = matrix_down.matrix();
#endif

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrix_up.regenerate();
 matrix_down.regenerate();
#endif

 dcomplex res = 0;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << k; // shifts the bits k times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : k) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.

  auto p = flip_index(matrix_up.get_x(nlc));
  matrix_up.change_one_row_and_one_col(nlc, nlc, p, p);
  matrix_down.change_one_row_and_one_col(nlc, nlc, p, p);

  res += sign * matrix_up.determinant() * matrix_down.determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res)))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
 }

#ifdef CHECK_GRAY_CODE_INTEGRITY
 double precision = 1.e-12;
 if (max_element(abs(mat_up - matrix_up.matrix())) > precision)
  TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_up - matrix_up.matrix()) << "Gray code: Not cyclic";
 if (max_element(abs(mat_do - matrix_down.matrix())) > precision)
  TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_do - matrix_down.matrix()) << "Gray code: Not cyclic";
 // check all indices of keldysh back to 0
 for (int n = 0; n < k; ++n) {
   if (matrix_up.get_x(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
   if (matrix_up.get_y(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
   if (matrix_down.get_x(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
   if (matrix_down.get_y(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
 }
#endif

 return res;
}
