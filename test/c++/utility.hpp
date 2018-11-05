#include <triqs/arrays.hpp>

using namespace triqs::arrays;
using dcomplex = std::complex<double>;

/**
 * Some comparison functions for tests.
 *
 * TODO: use it where needed
 */

template<typename T>
bool my_isfinite(T x) {return std::isfinite(x);};

bool my_isfinite(dcomplex z) {return (std::isfinite(z.real()) and std::isfinite(z.imag()));}

template<typename T>
bool is_close(T a, T b, double rtol, double atol) {
 return abs(a - b) <= rtol * abs(b) + atol;
};

template <typename T1, typename T2>
bool is_close_array(const T1& a, const T2& b, double rtol, double atol) {
 return max_element(abs(a - b) - rtol * abs(b)) <= atol;
};

template <typename T1, typename T2>
bool array_equal(const T1& a, const T2& b) {
 return max_element(abs(a - b)) == 0;
};

template <typename T>
bool all_positive(const T& a) {
 return array_equal(abs(a), a);
};

template <typename T>
bool all_finite(const T& a) {
 bool res = true;
 for (auto it = a.begin(); it != a.end(); ++it) {
  res = res and my_isfinite(*it);
 }
 return res;
};
