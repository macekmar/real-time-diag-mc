#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace moves {

// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {

 auto k = weight->perturbation_order; // order before adding a time
 quick_exit = (k >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new line and col.
 pt = get_random_point();
 for (auto &m : weight->matrices) m.insert(k, k, pt, pt);
 new_weight = recompute_sum_keldysh_indices(weight->matrices, k + 1);

 // The Metropolis ratio;
 return t_max_L_U / (k + 1) * new_weight / weight->value;
}

dcomplex insert::accept() {
 auto k = weight->perturbation_order;
 weight->value = new_weight;
 weight->perturbation_order++;
 measure->insert(k, pt);
 return 1.0;
}

void insert::reject() {
 if (quick_exit) return;
 auto k = weight->perturbation_order;
 for (auto &m : weight->matrices) m.remove(k, k);
}

// ------------ QMC double-insertion move --------------------------------------

dcomplex insert2::attempt() {
 auto k = weight->perturbation_order; // order before adding two times
 quick_exit = (k + 1 >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto pt1 = get_random_point();
 auto pt2 = get_random_point();
 for (auto &m : weight->matrices) m.insert2(k, k + 1, k, k + 1, pt1, pt2, pt1, pt2);

 new_weight = recompute_sum_keldysh_indices(weight->matrices, k + 2);

 // The Metropolis ratio
 return t_max_L_U * t_max_L_U / ((k + 1) * (k + 2)) * new_weight / weight->value;
}

dcomplex insert2::accept() {
 auto k = weight->perturbation_order;
 weight->value = new_weight;
 weight->perturbation_order += 2;
 measure->insert2(k, pt1, pt2);
 return 1.0;
}

void insert2::reject() {
 if (quick_exit) return;
 auto k = weight->perturbation_order;
 for (auto &m : weight->matrices) m.remove2(k, k + 1, k, k + 1);
}

//// ------------ QMC removal move --------------------------------------

dcomplex remove::attempt() {

 auto k = weight->perturbation_order; // order before removal
 quick_exit = (k <= params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                                                          // Choose one of the operators for removal
 removed_pt = weight->matrices[0].get_x(p);                           // store the point to be remove for later reject
 for (auto &m : weight->matrices) m.remove(p, p);                     // remove the point for all matrices
 new_weight = recompute_sum_keldysh_indices(weight->matrices, k - 1); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / t_max_L_U * new_weight / weight->value;
}

dcomplex remove::accept() {
 weight->value = new_weight;
 weight->perturbation_order--;
 measure->remove(p);
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 for (auto &m : weight->matrices) m.insert(p, p, removed_pt, removed_pt);
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {

 auto k = weight->perturbation_order; // order before removal
 quick_exit = (k - 2 < params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the operators for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // if remove p1, and p2 is later, ????? FIXME
 removed_pt1 = weight->matrices[0].get_x(p1);
 removed_pt2 = weight->matrices[0].get_x(p2);
 for (auto &m : weight->matrices) m.remove2(p1, p2, p1, p2);
 new_weight = recompute_sum_keldysh_indices(weight->matrices, k - 2); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(t_max_L_U, 2) * new_weight / weight->value;
}

dcomplex remove2::accept() {
 weight->value = new_weight;
 weight->perturbation_order -= 2;
 measure->remove2(p1, p2);
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 for (auto &m : weight->matrices) m.insert2(p1, p2, p1, p2, removed_pt1, removed_pt2, removed_pt1, removed_pt2);
}

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {

 auto k = weight->perturbation_order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                                // Choose one of the operators
 removed_pt = weight->matrices[0].get_x(p); // old time, to be saved for the removal case

 new_pt = get_random_point();                     // new time
 for (auto &m : weight->matrices) m.remove(p, p); // remove the point for all matrices
 for (auto &m : weight->matrices) m.insert(p, p, new_pt, new_pt);
 new_weight = recompute_sum_keldysh_indices(weight->matrices, k);

 // The Metropolis ratio
 return new_weight / weight->value;
}

dcomplex shift::accept() {
 weight->value = new_weight;
 measure->remove(p);
 measure->insert(p, new_pt);
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 for (auto &m : weight->matrices) m.remove(p, p); // remove the point for all matrices
 for (auto &m : weight->matrices) m.insert(p, p, removed_pt, removed_pt);
}

// ------------ QMC additional time swap move --------------------------------------

dcomplex weight_time_swap::attempt() {

 auto k = weight->perturbation_order;
 keldysh_contour_pt tau;

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                             // Choose one of the operators
 swap_pt = weight->matrices[0].get_x(p); // point whose time is to be swapped
 tau = weight->matrices[0].get_x(k);     // weight left point
 save_swap_pt = swap_pt;                 // save for reject case
 save_tau = tau;                         // save for reject case
 // swap normalised times:
 double tau_norm_time = (tau.t - weight->t_left_min) / (weight->t_left_max - weight->t_left_min);
 tau.t = weight->t_left_min + swap_pt.t * (weight->t_left_max - weight->t_left_min) / t_max;
 swap_pt.t = tau_norm_time * t_max;

 // replace points with swapped times
 for (auto &m : weight->matrices) m.change_one_row_and_one_col(p, p, swap_pt, swap_pt);
 weight->matrices[weight->op_to_measure_spin].change_row(k, tau);

 new_weight = recompute_sum_keldysh_indices(weight->matrices, k);

 // The Metropolis ratio
 return new_weight / weight->value;
}

dcomplex weight_time_swap::accept() {
 weight->value = new_weight;
 measure->change_one_row_and_one_col(p, p, swap_pt, swap_pt);
 return 1.0;
}

void weight_time_swap::reject() {
 if (quick_exit) return;
 auto k = weight->perturbation_order;
 for (auto &m : weight->matrices) m.change_one_row_and_one_col(p, p, save_swap_pt, save_swap_pt);
 weight->matrices[weight->op_to_measure_spin].change_row(k, save_tau);
}
}
