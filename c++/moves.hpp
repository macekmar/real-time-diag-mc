#pragma once
#include <triqs/mc_tools.hpp>
#include "./qmc_data.hpp"
#include "./parameters.hpp"

namespace moves {

struct common {
 qmc_data_t *data;
 const solve_parameters_t *params;
 triqs::mc_tools::random_generator &rng;
 double t_max_L_L_U = params->tmax * params->L * params->L * params->U;
 bool quick_exit = false;
 dcomplex sum_dets = 0;
 common(qmc_data_t *data, const solve_parameters_t *params, triqs::mc_tools::random_generator &rng)
    : data(data), params(params), rng(rng) {}
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
}
