#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace moves {

// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order; // order before adding a time
 quick_exit = (k >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new line and col.
 pt = get_random_point();
 integrand->weight->insert(k, pt);
 new_weight = integrand->weight->evaluate();

 // The Metropolis ratio;
 return t_max_L_U / (k + 1) * new_weight / integrand->weight->value;
}

dcomplex insert::accept() {
 auto k = integrand->perturbation_order;
 integrand->weight->value = new_weight;
 integrand->perturbation_order++;
 integrand->measure->insert(k, pt);
 return 1.0;
}

void insert::reject() {
 if (quick_exit) return;
 auto k = integrand->perturbation_order;
 integrand->weight->remove(k);
}

// ------------ QMC double-insertion move --------------------------------------

dcomplex insert2::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order; // order before adding two times
 quick_exit = (k + 1 >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto pt1 = get_random_point();
 auto pt2 = get_random_point();
 integrand->weight->insert2(k, k + 1, pt1, pt2);

 new_weight = integrand->weight->evaluate();

 // The Metropolis ratio
 return t_max_L_U * t_max_L_U / ((k + 1) * (k + 2)) * new_weight / integrand->weight->value;
}

dcomplex insert2::accept() {
 auto k = integrand->perturbation_order;
 integrand->weight->value = new_weight;
 integrand->perturbation_order += 2;
 integrand->measure->insert2(k, k + 1, pt1, pt2);
 return 1.0;
}

void insert2::reject() {
 if (quick_exit) return;
 auto k = integrand->perturbation_order;
 integrand->weight->remove2(k, k + 1);
}

//// ------------ QMC removal move --------------------------------------

dcomplex remove::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order; // order before removal
 quick_exit = (k <= params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                                    // Choose one of the operators for removal
 removed_pt = integrand->weight->get_config(p); // store the point to be remove for later reject
 integrand->weight->remove(p);                  // remove the point for all matrices
 new_weight = integrand->weight->evaluate();    // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / t_max_L_U * new_weight / integrand->weight->value;
}

dcomplex remove::accept() {
 integrand->weight->value = new_weight;
 integrand->perturbation_order--;
 integrand->measure->remove(p);
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 integrand->weight->insert(p, removed_pt);
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order; // order before removal
 quick_exit = (k - 2 < params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the operators for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // if remove p1, and p2 is later, ????? FIXME
 removed_pt1 = integrand->weight->get_config(p1);
 removed_pt2 = integrand->weight->get_config(p2);
 integrand->weight->remove2(p1, p2);
 new_weight = integrand->weight->evaluate(); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(t_max_L_U, 2) * new_weight / integrand->weight->value;
}

dcomplex remove2::accept() {
 integrand->weight->value = new_weight;
 integrand->perturbation_order -= 2;
 integrand->measure->remove2(p1, p2);
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 integrand->weight->insert2(p1, p2, removed_pt1, removed_pt2);
}

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                                    // Choose one of the operators
 removed_pt = integrand->weight->get_config(p); // old time, to be saved for the removal case

 new_pt = get_random_point(); // new time
 integrand->weight->change_config(p, new_pt);
 new_weight = integrand->weight->evaluate();

 // The Metropolis ratio
 return new_weight / integrand->weight->value;
}

dcomplex shift::accept() {
 integrand->weight->value = new_weight;
 integrand->measure->change_config(p, new_pt);
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 integrand->weight->change_config(p, removed_pt);
}

// ------------ QMC additional time swap move --------------------------------------

dcomplex weight_swap::attempt() {
  //integrand->weight->register_config();

 auto k = integrand->perturbation_order;
 keldysh_contour_pt tau;

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                                 // Choose one of the operators
 swap_pt = integrand->weight->get_config(p); // point whose time is to be swapped
 tau = integrand->weight->get_left_input();  // integrand left input point
 save_swap_pt = swap_pt;                     // save for reject case
 save_tau = tau;                             // save for reject case
 // swap times:
 double tau_time = tau.t;
 tau.t = swap_pt.t;
 tau.k_index = rng(2); // random keldysh index
 swap_pt.t = tau_time;

 // replace points with swapped times
 integrand->weight->change_config(p, swap_pt);
 integrand->weight->change_left_input(tau);

 new_weight = integrand->weight->evaluate();

 // The Metropolis ratio
 return new_weight / integrand->weight->value;
}

dcomplex weight_swap::accept() {
 integrand->weight->value = new_weight;
 integrand->measure->change_config(p, swap_pt);
 return 1.0;
}

void weight_swap::reject() {
 if (quick_exit) return;
 integrand->weight->change_config(p, save_swap_pt);
 integrand->weight->change_left_input(save_tau);
}

// ------------ QMC additional time shift move --------------------------------------

dcomplex weight_shift::attempt() {
 //integrand->weight->register_config();
 // No quick exit for this move, all orders are concerned

 keldysh_contour_pt tau = integrand->weight->get_left_input();  // integrand left input point
 save_tau = tau;                             // save for reject case
 tau.t = rng(physics_params->interaction_start + physics_params->t_max) - physics_params->interaction_start; // random time
 tau.k_index = rng(2); // random keldysh index
 integrand->weight->change_left_input(tau);

 new_weight = integrand->weight->evaluate();

 // The Metropolis ratio
 return new_weight / integrand->weight->value;
}

dcomplex weight_shift::accept() {
 integrand->weight->value = new_weight;
 return 1.0;
}

void weight_shift::reject() {
 // No quick exit for this move, all orders are concerned
 integrand->weight->change_left_input(save_tau);
}
}
