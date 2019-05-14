#pragma once
#include <vector>
#include <math.h>

/**
 * Spline polynomial interpolation with Bernstein polynomials
 *
 * Each object contains a 1D vector `intervals` of intervals limits of 
 * individual splines defined with `coeff`icients.
 *
 * Bernstein polynomials have the form
 * \[
 *   b_{\eta, n}(t) = \binom{n}{\eta} t^\eta (1-t)^(n-\eta)
 * \]
 * and the interpolation of the k-th spline is
 * \[
 *   p_k(t) = \sum_{\eta=0}^n c_{k,\eta} b_{\eta,n}(t),
 * \]
 * where $t \in [0, 1]$ is a normalized variable on the spline's interval.
 * 
 * In the importance sampling, we need the interpolation and its derivative, 
 * which can be calculated from equations above.
 * 
 * For each evaluation of $x$, we need to find the corresponding interval $k$. 
 * This is done with the bisection search. `intervals` have to be ordered.
 * 
 * Currently only the 3rd order polynomials are implemented. The class could be
 * generalized for different orders, but most probably we will never need them.
*/

class PolyInterp {
 private:
 std::vector<double> intervals;
 std::vector<std::vector<double>> coeff;
  
 inline double b02(double t) {return pow(1.0 - t, 2.0);};
 inline double b12(double t) {return 2.0*t*(1.0-t);};
 inline double b22(double t) {return pow(t, 2.0);};

 inline double b03(double t) {return pow(1.0 - t, 3.0);};
 inline double b13(double t) {return 3.0*t*pow(1.0 - t, 2.0);};
 inline double b23(double t) {return 3.0*pow(t, 2.0)*(1.0-t);};
 inline double b33(double t) {return pow(t, 3.0);};

 public:
 PolyInterp(){};
 PolyInterp(std::vector<double> intervals, std::vector<std::vector<double>> coeff): intervals(intervals), coeff(coeff) {};
 
 double get(double x);
 double get_derivative(double x);
 int get_interval(double x);

};