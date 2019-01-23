#include "./qmc_data.hpp"
#include "./configuration.hpp"

//#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE


void nice_print(det_manip<g0_keldysh_t> det, int p) {
 int n = det.size();
 std::cout << "DEBUG: x points = ";
 for (int i = 0; i < n; ++i) {
  std::cout << "(" << det.get_x(i).t << ", " << det.get_x(i).k_index << "), ";
 }
 std::cout << std::endl;
 std::cout << "DEBUG: y points = ";
 for (int i = 0; i < n; ++i) {
  if (i == p) std::cout << "              ";
  std::cout << "(" << det.get_y(i).t << ", " << det.get_y(i).k_index << "), ";
 }
 std::cout << std::endl;
 std::cout << std::endl;
}


// ---------------- Cofact * det ------------------
/// Gray code cofactor rotation.
// TODO: write down the formula this implements
// FIXME: not used ??
template <class T>
dcomplex Configuration<T>::keldysh_sum_cofact(int p) {

 if (order > 63) TRIQS_RUNTIME_ERROR << "order overflow";
 if (order < 1) TRIQS_RUNTIME_ERROR << "order cannot be zero";

 if (order == 1) {
  return matrices[0].determinant() * matrices[1].determinant();
 }

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[0].regenerate();
 matrices[1].regenerate();
#endif

 dcomplex res = 0;
 keldysh_contour_pt pt, pt_l, pt_r;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << order - 1; // shifts the bits order - 1 times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : order - 1) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.

  int nlc_p = nlc;

  nlc_p = nlc >= p ? nlc + 1 : nlc;


  pt_l = flip_index(matrices[0].get_x(nlc_p));
  pt_r = flip_index(matrices[0].get_y(nlc));
  // matrices[0].change_one_row_and_one_col(nlc_p, nlc, pt_l, pt_r);
  matrices[0].change_row(nlc_p, pt_l);
  matrices[0].change_col(nlc, pt_r);

  pt = flip_index(matrices[1].get_x(nlc_p));
  // matrices[1].change_one_row_and_one_col(nlc_p, nlc_p, pt, pt);
  matrices[1].change_row(nlc_p, pt);
  matrices[1].change_col(nlc_p, pt);

  // nice_print(matrices[0], p);
  res += sign * matrices[0].determinant() * matrices[1].determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res))))
   TRIQS_RUNTIME_ERROR << "NAN for n = " << n << ", res = " << res << ", order = " << order << ", p = " << p << ", nlc = " << nlc
                       << ", nlc_p = " << nlc_p << ", det0 = " << matrices[0].determinant()
                       << ", det1 = " << matrices[1].determinant();
 }

 return res;
}

// ---------------- det * det ------------------
/// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
template <class T>
dcomplex Configuration<T>::keldysh_sum() {

 if (order > 63) TRIQS_RUNTIME_ERROR << "order overflow";

 if (order == 0) {
  dcomplex res0 = matrices[0].determinant() * matrices[1].determinant();
  if (!(std::isfinite(real(res0)) & std::isfinite(imag(res0))))
   TRIQS_RUNTIME_ERROR << "NAN for n = 0, res = " << res0 << ", order = " << order;
  // return matrices[0].determinant() * matrices[1].determinant();
  return res0;
 }

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[0].regenerate();
 matrices[1].regenerate();
#endif

 dcomplex res = 0;
 keldysh_contour_pt pt;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << order; // shifts the bits order times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : order) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.

  for (auto spin : {up, down}) {
   pt = flip_index(matrices[spin].get_x(nlc));
   matrices[spin].change_row(nlc, pt);
   matrices[spin].change_col(nlc, pt);
  }

  res += sign * matrices[0].determinant() * matrices[1].determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res)))) {
   TRIQS_RUNTIME_ERROR << "NAN for n = " << n << ", res = " << res << ", order = " << order << ", nlc = " << nlc
                       << ", det0 = " << matrices[0].determinant() << ", det1 = " << matrices[1].determinant();
  }
 }

 return res;
}

/* Following are one matrix keldysh sums
// ---------------- Cofact ------------------
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t>& matrix, int k, int p) {

 if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";

 if (k == 0) {
  return matrix.determinant();
 }

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrix.regenerate();
#endif

 dcomplex res = 0;
 keldysh_contour_pt pt_l, pt_r;
 int nlc, nlc_l, nlc_r;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << k; // shifts the bits k times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : k) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.

  nlc_l = (nlc >= p) ? 2 * nlc + 2 : 2 * nlc;
  nlc_r = (nlc >= p) ? 2 * nlc + 1 : 2 * nlc;

  for (int v : {0, 1}) {
   pt_l = flip_index(matrix.get_x(nlc_l + v));
   pt_r = flip_index(matrix.get_y(nlc_r + v));
   matrix.change_one_row_and_one_col(nlc_l + v, nlc_r + v, pt_l, pt_r);
  }

  res += sign * matrix.determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res)))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
 }

 return res;
}


// ---------------- det ------------------
dcomplex recompute_sum_keldysh_indices(det_manip<g0_keldysh_t>& matrix, int k) {

 if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";

 if (k == 0) {
  return matrix.determinant();
 }

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrix.regenerate();
#endif

 dcomplex res = 0;
 keldysh_contour_pt pt_l, pt_r;
 int sign = -1;                    // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << k; // shifts the bits k times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : k) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
                                                  // to 1. ~n has bites inversed compared with n.
  nlc *= 2;

  for (int v : {0, 1}) {
   pt_l = flip_index(matrix.get_x(nlc + v));
   pt_r = flip_index(matrix.get_y(nlc + v));
   matrix.change_one_row_and_one_col(nlc + v, nlc + v, pt_l, pt_r);
  }

  res += sign * matrix.determinant();
  sign = -sign;

  if (!(std::isfinite(real(res)) & std::isfinite(imag(res)))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
 }

 return res;
}
*/

template class Configuration<ConfigurationQMC>;
template class Configuration<ConfigurationAuxMC>;
