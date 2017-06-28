#pragma once
#include "./measure.hpp"
#include "./parameters.hpp"
#include "./qmc_data.hpp"

namespace moves {

struct common {
 Configuration& config;
 const solve_parameters_t *params;
 triqs::mc_tools::random_generator &rng;
 random_x_generator rxg;
 double delta_t_L_U;
 double delta_t;
 bool quick_exit = false;

 common(Configuration* config, const solve_parameters_t *params, const double t_max,
        triqs::mc_tools::random_generator &rng)
    : config(*config), params(params), rng(rng) {
  rxg = random_x_generator();
  delta_t = params->interaction_start + t_max;
  delta_t_L_U = delta_t * rxg.size() * params->U;
 }

 /// Construct random point with space/orbital index, time and alpha
 keldysh_contour_pt get_random_point() {
  return {rxg(rng),
          qmc_time_t{rng(delta_t) - params->interaction_start}, 0};
 }
};

// ------------ QMC insertion move --------------------------------------

struct insert : common {

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

// ------------ QMC double-insertion move --------------------------------------

struct insert2 : common {

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
 int p;

 using common::common;
 dcomplex attempt();
 dcomplex accept();
 void reject();
};

}
