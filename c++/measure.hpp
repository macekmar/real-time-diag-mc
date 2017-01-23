#pragma once
#include "./qmc_data.hpp"
#include "./weight.hpp"
#include <triqs/arrays.hpp>
#include <triqs/clef.hpp>

using triqs::det_manip::det_manip;


// --------------- Measure ----------------

class Measure {

 protected:
 array<dcomplex, 1> value; // measure of the last accepted config
 double singular_threshold = 1e-12; // for set_manip. Not ideal to be defined here

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
 Weight* weight;
 Measure* measure;

 Integrand(Weight* weight, Measure* measure) : weight(weight), measure(measure){};
};

// ----------------

class weight_sign_measure : public Measure {

 const Weight* weight;

 weight_sign_measure(const weight_sign_measure&) = delete; // non construction-copyable
 void operator=(const weight_sign_measure&) = delete;      // non copyable

 public:
 weight_sign_measure(const input_physics_data* physics_params, const Weight* weight);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};

// ---------------

class twodet_single_measure : public Measure {

 private:
 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down with tau and tau'
 const input_physics_data* physics_params;

 twodet_single_measure(const twodet_single_measure&) = delete; // non construction-copyable
 void operator=(const twodet_single_measure&) = delete;        // non copyable

 public:
 twodet_single_measure(const input_physics_data* physics_params);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};

// ---------------

class twodet_multi_measure : public Measure {

 private:
 std::vector<det_manip<g0_keldysh_t>>
     matrices; // M matrices for up and down without tau and tau', so they are actually the same...
 const input_physics_data* physics_params;

 twodet_multi_measure(const twodet_multi_measure&) = delete; // non construction-copyable
 void operator=(const twodet_multi_measure&) = delete;       // non copyable

 public:
 twodet_multi_measure(const input_physics_data* physics_params);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};

// ---------------

class twodet_cofact_measure : public Measure {

 private:
 std::vector<det_manip<g0_keldysh_t>>
     matrices; // M matrices for up and down without tau and tau', so they are actually the same...
 g0_keldysh_t green_function;
 const input_physics_data* physics_params;

 twodet_cofact_measure(const twodet_cofact_measure&) = delete; // non construction-copyable
 void operator=(const twodet_cofact_measure&) = delete;        // non copyable

 public:
 twodet_cofact_measure(const input_physics_data* physics_params);

 void insert(int k, keldysh_contour_pt pt);
 void insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2);
 void remove(int k);
 void remove2(int k1, int k2);
 void change_config(int k, keldysh_contour_pt pt);
 void evaluate();
};
