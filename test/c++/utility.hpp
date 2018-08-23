#include <triqs/arrays.hpp>

using namespace triqs::arrays;
using dcomplex = std::complex<double>;

/**
 * Some comparison functions for tests.
 *
 * TODO: use it where needed
 */

template<typename T>
bool is_close(T a, T b, double rtol, double atol) {
 return abs(a - b) <= rtol * abs(b) + atol;
};

template <typename T, int D>
bool is_close_array(array<T, D>& a, array<T, D>& b, double rtol, double atol) {
 return max_element(abs(a - b) - rtol * abs(b)) <= atol;
};

