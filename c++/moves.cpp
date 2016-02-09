#include "./moves.hpp"
#include <triqs/det_manip.hpp>
#include "./keldysh_sum.hpp"

namespace moves {

keldysh_contour_pt get_random_point(triqs::mc_tools::random_generator &rng, const solve_parameters_t *params) {
 return {get_random_x(rng,params), qmc_time_t{rng(params->tmax)}, 0};
}
// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {

 auto k = data->perturbation_order; // order before adding a time
 quick_exit = (k >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new line and col.
 auto p = get_random_point(rng, params);
 for (auto &m : data->matrices) m.insert(k, k, p, p);
 sum_dets = recompute_sum_keldysh_indices(data, k + 1);

 // The Metropolis ratio
 // FIXME: t_max_L_L_U : dans le get_random_point policy
 return t_max_L_L_U / (k + 1) * sum_dets / data->sum_keldysh_indices;
}

dcomplex insert::accept() {
 data->sum_keldysh_indices = sum_dets;
 data->perturbation_order++;
 return 1.0;
}

void insert::reject() {
 if (quick_exit) return;
 auto k = data->perturbation_order;
 for (auto &m : data->matrices) m.remove(k, k);
}

// ------------ QMC double-insertion move --------------------------------------

dcomplex insert2::attempt() {
 auto k = data->perturbation_order; // order before adding two times
 quick_exit = (k + 2 > params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto p1 = get_random_point(rng, params);
 auto p2 = get_random_point(rng, params);
 for (auto &m : data->matrices) m.insert2(k, k + 1, k, k + 1, p1, p2, p1, p2);

 sum_dets = recompute_sum_keldysh_indices(data, k + 2);

 // The Metropolis ratio
 return t_max_L_L_U * t_max_L_L_U / ((k + 1) * (k + 2)) * sum_dets / data->sum_keldysh_indices;
}

dcomplex insert2::accept() {
 data->sum_keldysh_indices = sum_dets;
 data->perturbation_order += 2;
 return 1.0;
}

void insert2::reject() {
 if (quick_exit) return;
 auto k = data->perturbation_order;
 for (auto &m : data->matrices) m.remove2(k, k + 1, k, k + 1);
}


// ------------ QMC removal move --------------------------------------

dcomplex remove::attempt() {

 auto k = data->perturbation_order; // order before removal
 quick_exit = (k <= params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                                            // Choose one of the operators for removal
 removed_pt = data->matrices[0].get_x(p);               // store the point to be remove for later reject
 for (auto &m : data->matrices) m.remove(p, p);         // remove the point for all matrices
 sum_dets = recompute_sum_keldysh_indices(data, k - 1); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / t_max_L_L_U * sum_dets / data->sum_keldysh_indices;
}

dcomplex remove::accept() {
 data->sum_keldysh_indices = sum_dets;
 data->perturbation_order--;
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 for (auto &m : data->matrices) m.insert(p, p, removed_pt, removed_pt);
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {

 auto k = data->perturbation_order; // order before removal
 quick_exit = (k - 2 < params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the operators for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // if remove p1, and p2 is later, ????? FIXME
 removed_pt1 = data->matrices[0].get_x(p1);
 removed_pt2 = data->matrices[0].get_x(p2);
 for (auto &m : data->matrices) m.remove2(p1, p2, p1, p2);
 sum_dets = recompute_sum_keldysh_indices(data, k - 2); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(t_max_L_L_U, 2) * sum_dets / data->sum_keldysh_indices;
}

dcomplex remove2::accept() {
 data->sum_keldysh_indices = sum_dets;
 data->perturbation_order -= 2;
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 for (auto &m : data->matrices) m.insert2(p1, p2, p1, p2, removed_pt1, removed_pt2, removed_pt1, removed_pt2);
}
}
