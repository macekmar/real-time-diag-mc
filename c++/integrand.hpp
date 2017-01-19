#pragma once
#include "./measure.hpp"
#include "./weight.hpp"

class Integrand {
 // Represents everything under the QMC integral. Depends on the QMC configuration which is a list of n alphas (n is the
 // perturbation order), and on the inputs \tau and \tau'. In the general case, each alpha is a 2x2 matrix of keldysh point (site,
 // time, keldysh index). Depending on symmetries it can be simplified to a single keldysh point. The integrand can be separated
 // in a weight and a measure, or not. If so, changes in the weight and the measure are separated to improve calculation time.

 public:
 int perturbation_order = 0; // managed by the moves
 Weight* weight;             // used publicly by the moves

 Integrand() : weight(NULL){};
 Integrand(Weight* weight) : weight(weight){};

 virtual array<dcomplex, 1> get_measure_value() = 0;
 virtual void measure_evaluate() = 0;
 virtual void measure_insert(int k, keldysh_contour_pt pt) = 0;
 virtual void measure_insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) = 0;
 virtual void measure_remove(int k) = 0;
 virtual void measure_remove2(int k1, int k2) = 0;
 virtual void measure_change_config(int k, keldysh_contour_pt pt) = 0;
};

// ------------------

class single_block_integrand : public Integrand {

 public:
 single_block_integrand(Weight* weight) : Integrand(weight){};

 array<dcomplex, 1> get_measure_value(){}; // must raise some error

 // for the following, do nothing (work has already been done in weight, but these functions are still called):
 void measure_evaluate(){};
 void measure_insert(int k, keldysh_contour_pt pt){};
 void measure_insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2){};
 void measure_remove(int k){};
 void measure_remove2(int k1, int k2){};
 void measure_change_config(int k, keldysh_contour_pt pt){};
};

class separated_integrand : public Integrand {

 private:
 Measure* measure;

 public:
 separated_integrand(Weight* weight, Measure* measure) : Integrand(weight), measure(measure){};

 array<dcomplex, 1> get_measure_value() { return measure->get_value(); };
 void measure_evaluate() { measure->evaluate(); };
 void measure_insert(int k, keldysh_contour_pt pt) { measure->insert(k, pt); };
 void measure_insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) { measure->insert2(k1, k2, pt1, pt2); };
 void measure_remove(int k) { measure->remove(k); };
 void measure_remove2(int k1, int k2) { measure->remove2(k1, k2); };
 void measure_change_config(int k, keldysh_contour_pt pt) { measure->change_config(k, pt); };
};
