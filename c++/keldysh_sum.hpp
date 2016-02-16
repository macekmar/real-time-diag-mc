#pragma once

#define CHECK_GRAY_CODE_INTEGRITY
#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

/// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
dcomplex recompute_sum_keldysh_indices(qmc_data_t* data, int k) {

 // int k = perturbation_order();
 if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";
 auto& matrices = data->matrices;

 // When no time is inserted, only the observable is present in the matrix
 // if (k == 0) return imag(g0_lesser(mindex(0,0,0),0));

 //if (k == 0) return imag(matrices[up].determinant() * matrices[down].determinant());
 
 //std::cout << "Green function  = " << imag(data->g0_lesser_0) << std::endl;
 if (k == 0) return imag(data->g0_lesser_0); //FIXME hardcoded for the moment

 // FIXME From Laura's code
 // When no time is inserted, only the observable is present in the matrix
 // measure i<d^+ c_k> for the current, but <d^+ d> for charge.
 //if (N == 0) return current ? real(g0_equal_time()) : imag(g0_equal_time());

#ifdef CHECK_GRAY_CODE_INTEGRITY
 auto mat_up = matrices[up].matrix();
 auto mat_do = matrices[down].matrix();
#endif

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[up].regenerate();
 matrices[down].regenerate();
#endif

 dcomplex res = 0;
 int sign = -1; // FIXME WHY IS THIS -1 ??
 auto two_to_k = uint64_t(1) << k; // shifts the bits k time to the left
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

  if (!std::isfinite(real(res))) TRIQS_RUNTIME_ERROR << "NAN for n = " << n;
 }

 dcomplex i_n[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // powers of i
 res = -res * i_n[(k + 1) % 4];                        // * i^(k+1) //FIXME -sign before res?!

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

 return real(res); // FIXME :
}

////FIXME From Laura's code
//  //---------------------------------------------------------------
// /// Gray code determinant rotation. Returns the sum of prod of det for all keldysh configurations.
// dcomplex recompute_sum_keldysh_indices(qmc_data_t* data, int k) {
//  if (k > 63) TRIQS_RUNTIME_ERROR << "k overflow";
//
//  //When no time is inserted, only the observable is present in the matrix
//  if (k == 0)
//    return current?real(g0_equal_time()):imag(g0_equal_time()); // measure i<d^+ c_k> for the current, but <d^+ d> for charge.
//    
//  auto two_to_N = uint64_t(1) << k; //shifts he bites from k to the left
//  auto R_down = range(0,k);
//  auto R_up   = range(0,k+1);
//  dcomplex res = 0;
//  int sign = -1;
//  
//  // first construction of the matrices
//  for (size_t i=0; i<k+1;i++)
//    for (size_t j=0; j<k+1;j++)
//      matrices[up](i,j) = g0bar_t(x_values[up][i], y_values[up][j]);
//    for (size_t i=0; i<k;i++)
//      for (size_t j=0; j<k;j++)
//        matrices[down](i,j) = g0bar_t(x_values[down][i], y_values[down][j]);
//
//  for (uint64_t n = 0; n < two_to_N; ++n) {
//
//   // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
//   int nlc = (n < two_to_N - 1 ? ffs(~n) : k) - 1; // ffs starts at 1, returns the position of the 1st (least significant) bit set to 1. ~n has bites inversed compared with n.
//
//   auto t = flip_index(x_values[down][nlc]);
//   x_values[up][nlc+1]=t;
//   y_values[up][nlc+1]=t;
//   x_values[down][nlc]=t;
//   y_values[down][nlc]=t;
//   for (size_t i=0; i<k+1;i++){
//     matrices[up](i,nlc+1) = g0bar_t(x_values[up][i], t);
//     matrices[up](nlc+1,i) = g0bar_t(t, y_values[up][i]);
//   }
//   for (size_t i=0; i<k;i++){
//     matrices[down](i,nlc) = g0bar_t(x_values[down][i], t);
//     matrices[down](nlc,i) = g0bar_t(t, y_values[down][i]);
//   }
//   
//   res += sign * determinant(matrices[up](R_up,R_up)) * determinant(matrices[down](R_down,R_down));
//   sign *= -1;
//
//  }
//  if (!std::isfinite(real(res))) TRIQS_RUNTIME_ERROR << "NAN ";
//
//  res = - res * i_n[(k + 1)%4]; // * i^(k+1)
//  if (current) res *= 1_j; // measure i<d^+ c_k> for the current, but <d^+ d> for charge.
//
//  return real(res);
// }
