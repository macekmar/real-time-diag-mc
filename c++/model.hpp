#pragma once
#include "./poly_interp.hpp"
#include "./qmc_data.hpp"
#include <vector>

/**
 * `Model` implements the model for the importance sampling
 * 
 * Our model for f(u) is a product of 1D functions 
 * $f_i(\bar{u}_i - \bar{u}_{i-1})$, where $\bar{u}_i$ are ordered 
 * $u = [u_1 u_2 \dots u_n]$ such that
 * \[
 *   t_{int. start} <= \bar{u}_n <= \bar{u}_{n-1} <= \dots <= \bar{u}_1 <= 0
 * \]
 * The seperable variables are
 * \[
 *   v_i = \bar{u}_i - \bar{u}{i-1}, \quad i = 1, \dots, n,
 * \]
 * where $\bar{u}_0 = 0$.
 * 
 * For importance sampling we have to provide a list of $F_i^{-1}(l)$.
 * Each $F_i^{-1}(l)$ is a cubic spline interpolation $p_i(l)$ described by the 
 * interval splits of $l$ for different splines $p_{i,k}(l)$[1] and their 
 * coefficitens $c_{k,\eta}, \eta = 0,1,2,3$.
 * 
 * How to calculate $f(u)$?
 * Originally, $f_i$ were just integrands of the first order $W_1$, so we
 * could use them. However, $f_i$ has to be derived from the same function as 
 * $F_i^{-1}(l)$. Using the derivative of an inverse, we get
 * \[
 *    f_i(\bar{u}_i) = 1/p_i'(l).
 * \]
 * Since $p_i$ is a polynomial, its derivative is easy to calculate.
 * 
 */

class Model {

 private:
 std::vector<PolyInterp> interpolators;

 public:
 Model(){};
 Model(std::vector<std::vector<double>> intervals, std::vector<std::vector<std::vector<double>>> coeff);

 dcomplex weight;

 std::vector<timec_t> l_to_v(std::vector<timec_t> l);
 std::vector<timec_t> v_to_u(std::vector<timec_t> v);
 std::vector<timec_t> l_to_u(std::vector<timec_t> l);
 std::vector<timec_t> u_to_v(std::vector<timec_t> u);
 std::vector<timec_t> v_to_l(std::vector<timec_t> v);
 std::vector<timec_t> u_to_l(std::vector<timec_t> u);

 timec_t inverse_cdf (int i, timec_t l);
 timec_t cdf (int i, timec_t v);
 void evaluate(std::vector<double> times);
};