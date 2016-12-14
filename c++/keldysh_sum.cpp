#include "./qmc_data.hpp"

//#define CHECK_GRAY_CODE_INTEGRITY
#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

/// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
dcomplex recompute_sum_keldysh_indices(qmc_data_t* data, const solve_parameters_t* params, int k) {

 if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";
 auto& matrices = data->matrices;

 if (k == 0) {
  return matrices[up].determinant() * matrices[down].determinant();
 }

#ifdef CHECK_GRAY_CODE_INTEGRITY
 auto mat_up = matrices[up].matrix();
 auto mat_do = matrices[down].matrix();
#endif

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[up].regenerate();
 matrices[down].regenerate();
#endif

 dcomplex res = 0;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << k; // shifts the bits k times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : k) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.

  auto p = flip_index(matrices[0].get_x(nlc));
  matrices[up].change_one_row_and_one_col(nlc, nlc, p, p);
  matrices[down].change_one_row_and_one_col(nlc, nlc, p, p);

  res += sign * matrices[up].determinant() * matrices[down].determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res)))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
 }

#ifdef CHECK_GRAY_CODE_INTEGRITY
 double precision = 1.e-12;
 if (max_element(abs(mat_up - matrices[up].matrix())) > precision)
  TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_up - matrices[up].matrix()) << "Gray code: Not cyclic";
 if (max_element(abs(mat_do - matrices[down].matrix())) > precision)
  TRIQS_RUNTIME_ERROR << matrix<dcomplex>(mat_do - matrices[down].matrix()) << "Gray code: Not cyclic";
 // check all indices of keldysh back to 0
 for (int n = 0; n < k; ++n) {
  for (auto const& mat : matrices) {
   if (mat.get_x(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
   if (mat.get_y(n).k_index != 0) TRIQS_RUNTIME_ERROR << "Gray code: Keldysh index is not 0 !!";
  }
 }
#endif

 return res;
}
