#pragma once
#include "./poly_interp.hpp"
#include <triqs/utility/exceptions.hpp>
#include <iostream>

/**
 * Finds the correct index (interval) for x
 * It uses bisecton, `intervals` should be ordered
 */
int PolyInterp::get_interval(double x) {
 int bottom = 0;
 int top = intervals.size()-1;
 int center = bottom + (top - bottom)/2;
 
 while ( ((center - bottom) > 0) && ((top - center) > 0)) {
  if (intervals[center] > x)
   top = center;
  else
   bottom = center;
  center = bottom + (top - bottom)/2;
 }

 if (x < intervals[center]) 
  TRIQS_RUNTIME_ERROR << "Bisection found wrong interval for the polynomial interpolation";
 if (x > intervals[center+1]) 
  TRIQS_RUNTIME_ERROR << "Bisection found wrong interval for the polynomial interpolation";
 
 return center;
}

/**
 * Returns the polynomial interpolation in x
 */
double PolyInterp::get(double x) {
 int i = get_interval(x);
 double t = (x - intervals[i])/(intervals[i+1] - intervals[i]);

 return coeff[i][0]*b03(t) + coeff[i][1]*b13(t) + coeff[i][2]*b23(t) + coeff[i][3]*b33(t);
}

/**
 * Returns the derivative of the polynomial interpolation in x
 */
double PolyInterp::get_derivative(double x) {
 int i = get_interval(x);
 double dx = (intervals[i+1] - intervals[i]);
 double t = (x - intervals[i])/dx;

 double c0 = 3.0*(coeff[i][1] - coeff[i][0])/dx;
 double c1 = 3.0*(coeff[i][2] - coeff[i][1])/dx;
 double c2 = 3.0*(coeff[i][3] - coeff[i][2])/dx;

 return c0*b02(t) + c1*b12(t) + c2*b22(t);
}
