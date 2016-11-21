#pragma once
#include "./parameters.hpp"
#include "./qmc_data.hpp"

namespace moves {

struct common {
 qmc_data_t *data;
 const solve_parameters_t *params;
 triqs::mc_tools::random_generator &rng;
 random_x_generator rxg;
 double t_max_L_U;
 bool quick_exit = false;
 dcomplex sum_dets = 0;

 common(qmc_data_t *data, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
    : data(data), params(params), rng(rng), rxg{data->matrices[0].get_function().g0_lesser.g0, params} {
  t_max_L_U = data->tmax * rxg.size() * params->U;
 }

 /// Construct random point with space/orbital index, time and alpha
 keldysh_contour_pt get_random_point() { return {rxg(rng), qmc_time_t{rng(data->tmax)}, 0}; }
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
