#pragma once
#include "./qmc_data.hpp"
#include "./weight.hpp"
#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>

using triqs::det_manip::det_manip;


// --------------- Measure ----------------

class Measure {

 protected:
 array<dcomplex, 1> value;         // measure of the last accepted config
 double singular_threshold = 1e-4; // for det_manip. Not ideal to be defined here

 public:
 array<dcomplex, 1> get_value() { return value; };
 virtual void evaluate() = 0;
 virtual void insert(int k, keldysh_contour_pt pt) = 0;
 virtual void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) = 0;
 virtual void remove(int k) = 0;
 virtual void remove2(int k1, int k2) = 0;
 virtual void change_config(int k, keldysh_contour_pt pt) = 0;
};

// ------------------------

struct Integrand {
 // Represents everything under the QMC integral. Depends on the QMC configuration which is a list of n alphas (n is the
 // perturbation order), and on the inputs \tau and \tau'. In the general case, each alpha is a 2x2 matrix of keldysh point (site,
 // time, keldysh index). Depending on symmetries it can be simplified to a single keldysh point.

 int perturbation_order = 0;
 Weight* weight = nullptr;
 Measure* measure = nullptr;

 // Integrand(const Integrand&) = delete; // non construction-copyable
 // void operator=(const Integrand&) = delete;      // non copyable

 Integrand(){};
 Integrand(Weight* weight, Measure* measure) : weight(weight), measure(measure){};
};

// ----------------

class weight_sign_measure : public Measure {

 const Weight* weight;

 weight_sign_measure(const weight_sign_measure&) = delete; // non construction-copyable
 void operator=(const weight_sign_measure&) = delete;      // non copyable

 public:
 weight_sign_measure(const Weight* weight);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};

// ---------------
// interaction between two distinct particles
class twodet_cofact_measure : public Measure {

 private:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down without tau and tau'
 g0_keldysh_t green_function;
 const std::vector<keldysh_contour_pt>* tau_list;
 const keldysh_contour_pt taup;
 const array<dcomplex, 1>* g0_values;
 const int op_to_measure_spin;

 twodet_cofact_measure(const twodet_cofact_measure&) = delete; // non construction-copyable
 void operator=(const twodet_cofact_measure&) = delete;        // non copyable

 public:
 twodet_cofact_measure(g0_keldysh_t green_function, const std::vector<keldysh_contour_pt>* tau_list,
                       const keldysh_contour_pt taup, const int op_to_measure_spin, const array<dcomplex, 1>* g0_values);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};

// ---------------
// interaction between two symmetric distinct particles
class onedet_cofact_measure : public Measure {

 private:
 det_manip<g0_keldysh_t> matrix; // in this case the two deterinants without tau and taup are the same
 g0_keldysh_t green_function;

 onedet_cofact_measure(const onedet_cofact_measure&) = delete; // non construction-copyable
 void operator=(const onedet_cofact_measure&) = delete;        // non copyable

 public:
 onedet_cofact_measure();

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};
