#include "./model.hpp"

Model::Model(std::vector<std::vector<double>> intervals, std::vector<std::vector<std::vector<double>>> coeff) {
  if (intervals.size() != coeff.size()) {
  TRIQS_RUNTIME_ERROR << "Shape mismatch between model in intervals and coefficients.";
 }

 for(int i = 0; i < intervals.size(); i++) {
  interpolators.push_back(PolyInterp(intervals[i], coeff[i]));
 }
}

// ============================================================================

std::vector<timec_t> Model::l_to_v(std::vector<timec_t> l) {
 if (l.size() > interpolators.size())
 {
  TRIQS_RUNTIME_ERROR << "Demanded order is too high for the model.";
 }
 std::vector<timec_t> v;

 int i = 0;
 for (auto t = l.begin(); t != l.end(); ++t) {
  v.push_back(inverse_cdf(i, *t));
  ++i;
 }
 return v;
}

std::vector<timec_t> Model::v_to_u(std::vector<timec_t> v) {
 std::vector<timec_t> u;
 u.push_back(-v.back());
 if (v.size() > 1) {
  for (auto t = std::prev(v.end(), 2); t != v.begin(); --t) {
    u.emplace_back(u.back() - *t);
  }
  u.emplace_back(u.back() - v.front());
  std::reverse(u.begin(), u.end());
 }
 return u;
}

std::vector<timec_t> Model::l_to_u(std::vector<timec_t> l) {
 return v_to_u(l_to_v(l));
}

// Reverse

std::vector<timec_t> Model::u_to_v(std::vector<timec_t> u) {
 std::sort(u.begin(), u.end(), std::less<timec_t>());
 std::vector<timec_t> v;
 auto prev = u.begin();
  for (auto t = std::next(u.begin()); t != u.end(); ++t) {
  v.push_back(*t - *prev);
  prev = t;
 }
 v.push_back(-*std::prev(u.end()));
 return v;
}

std::vector<timec_t> Model::v_to_l(std::vector<timec_t> v) {
if (v.size() > interpolators.size())
 {
  TRIQS_RUNTIME_ERROR << "Demanded order is too high for the model.";
 }
 std::vector<timec_t> l;
 int i = 0;
 for (auto t = v.begin(); t != v.end(); ++t) {
  l.push_back(cdf(i, *t));
  ++i;
 }
 return l;
}

std::vector<timec_t> Model::u_to_l(std::vector<timec_t> u) {
 return v_to_l(u_to_v(u));
}

// ============================================================================

timec_t Model::inverse_cdf(int i, timec_t l) {
 return interpolators[i].get(l); 
}

/**
 * We could provide the interpolation for CDF besides the interpolation for 
 * CDF^-1, but this would be a different function.
 * Instead, we have to solve a cubic equation (polynomials in the interpolation 
 * are cubic).
 * There exists an analytical solution for the cubic equations, but I find it
 * easier to do a bisection. Bisection will also always work, since the 
 * interpolation is monotonic by construction.
 */
timec_t Model::cdf(int i, timec_t v) {
 timec_t l_bottom = 0;
 timec_t l_top = 1.0;
 timec_t l_center = (l_bottom + l_top)/2;
 timec_t v_center = interpolators[i].get(l_center);

 if (interpolators[i].get(l_top) < v) {
  TRIQS_RUNTIME_ERROR << "v-value is above the interpolation range.";
 }
 if (interpolators[i].get(l_bottom) > v) {
  TRIQS_RUNTIME_ERROR << "v-value is below the interpolation range.";
 }

 while (l_top - l_bottom > 1e-15) {
  if ((v_center) > v)
   l_top = l_center;
  else
   l_bottom = l_center;
   
  l_center = (l_top + l_bottom)/2;
  v_center = interpolators[i].get(l_center);
 }

 return l_center;
}

/**
 * Calculates the model weight 
 */
void Model::evaluate(std::vector<double> times_l) {

 weight = 1.0;
 
 if (times_l.size() > interpolators.size())
 {
  TRIQS_RUNTIME_ERROR << "Demanded order is too high for the model.";
 }

 int i = 0;
 for (auto l = times_l.begin(); l != times_l.end(); ++l) {
  weight *= (dcomplex) (1.0/interpolators[i].get_derivative(*l));
  i++;
 }
}