#pragma once
#include "./measure.hpp"
#include "./integrand.hpp"
#include "./parameters.hpp"
#include "./qmc_data.hpp"
#include "./weight.hpp"

namespace moves {

struct common {
 Integrand *integrand;
 const solve_parameters_t *params;
 input_physics_data* physics_params;
 triqs::mc_tools::random_generator &rng;
 random_x_generator rxg;
 double t_max_L_U;
 bool quick_exit = false;
 dcomplex new_weight = 0;

 common(Integrand *integrand, const solve_parameters_t *params, const input_physics_data* physics_params, triqs::mc_tools::random_generator &rng)
    : integrand(integrand), params(params), rng(rng), rxg{} {
  t_max_L_U = physics_params->t_max * rxg.size() * params->U;
 }

 /// Construct random point with space/orbital index, time and alpha
 keldysh_contour_pt get_random_point() { return {rxg(rng), qmc_time_t{rng(physics_params->t_max)}, 0}; }
};

// ------------ QMC insertion move --------------------------------------

struct insert : common {
 keldysh_contour_pt pt;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-insertion move --------------------------------------

struct insert2 : common {
 keldysh_contour_pt pt1, pt2;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC removal move --------------------------------------

struct remove : common {
 keldysh_contour_pt removed_pt;
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-removal move --------------------------------------

struct remove2 : common {
 keldysh_contour_pt removed_pt1, removed_pt2;
 int p1, p2;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

//-----------QMC vertex shift move------------

struct shift : common {
 keldysh_contour_pt removed_pt;
 keldysh_contour_pt new_pt;
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

//-----------QMC additional time swap move------------

struct weight_time_swap : common {
 keldysh_contour_pt save_swap_pt, swap_pt;
 keldysh_contour_pt save_tau;
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};
}
