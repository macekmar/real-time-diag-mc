#include <triqs/det_manip.hpp>
#include "./moves.hpp"

namespace moves {

// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {

 auto k = data->perturbation_order; // order before adding a time
 quick_exit = (k >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new line and col.
 auto p = get_random_point();
 for (auto &m : data->matrices) m.insert(k, k, p, p);
 sum_dets = recompute_sum_keldysh_indices(data, params, k + 1);

 // The Metropolis ratio;
 return t_max_L_U / (k + 1) * sum_dets / data->sum_keldysh_indices;
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
 auto p1 = get_random_point();
 auto p2 = get_random_point();
 for (auto &m : data->matrices) m.insert2(k, k + 1, k, k + 1, p1, p2, p1, p2);

 sum_dets = recompute_sum_keldysh_indices(data, params, k + 2);

 // The Metropolis ratio
 return t_max_L_U * t_max_L_U / ((k + 1) * (k + 2)) * sum_dets / data->sum_keldysh_indices;
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
 p = rng(k);                                                    // Choose one of the operators for removal
 removed_pt = data->matrices[0].get_x(p);                       // store the point to be remove for later reject
 for (auto &m : data->matrices) m.remove(p, p);                 // remove the point for all matrices
 sum_dets = recompute_sum_keldysh_indices(data, params, k - 1); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / t_max_L_U * sum_dets / data->sum_keldysh_indices;
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
 sum_dets = recompute_sum_keldysh_indices(data, params, k - 2); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(t_max_L_U, 2) * sum_dets / data->sum_keldysh_indices;
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

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {

 auto k = data->perturbation_order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                              // Choose one of the operators
 removed_pt = data->matrices[0].get_x(p); // old time, to be saved for the removal case

 auto new_pt = get_random_point();              // new time
 for (auto &m : data->matrices) m.remove(p, p); // remove the point for all matrices
 for (auto &m : data->matrices) m.insert(p, p, new_pt, new_pt);
 sum_dets = recompute_sum_keldysh_indices(data, params, k);

 // The Metropolis ratio
 return sum_dets / data->sum_keldysh_indices;
}

dcomplex shift::accept() {
 data->sum_keldysh_indices = sum_dets;
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 for (auto &m : data->matrices) m.remove(p, p); // remove the point for all matrices
 for (auto &m : data->matrices) m.insert(p, p, removed_pt, removed_pt);
}
}
