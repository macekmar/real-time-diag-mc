#include "./moves.hpp"
#include <triqs/det_manip.hpp>

//#define REGISTER_CONFIG

namespace moves {

// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order; // order before adding a time
 quick_exit = (k >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new line and col.
 auto pt = get_random_point();
 config.insert(k, pt);
 new_weight = config.weight_evaluate();

 // The Metropolis ratio;
 return delta_t_L_U / (k + 1) * new_weight / config.weight_value;
}

dcomplex insert::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void insert::reject() {
 if (quick_exit) return;
 config.remove(config.order - 1);
}

// ------------ QMC double-insertion move --------------------------------------

dcomplex insert2::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order; // order before adding two times
 quick_exit = (k + 1 >= params->max_perturbation_order);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto pt1 = get_random_point();
 auto pt2 = get_random_point();
 config.insert2(k, k + 1, pt1, pt2);

 new_weight = config.weight_evaluate();

 // The Metropolis ratio
 return delta_t_L_U * delta_t_L_U / ((k + 1) * (k + 2)) * new_weight / config.weight_value;
}

dcomplex insert2::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void insert2::reject() {
 if (quick_exit) return;
 auto k = config.order;
 config.remove2(k - 2, k - 1);
}

//// ------------ QMC removal move --------------------------------------

dcomplex remove::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order; // order before removal
 quick_exit = (k <= params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                            // Choose one of the operators for removal
 removed_pt = config.get_config(p);     // store the point to be remove for later reject
 config.remove(p);                      // remove the point for all matrices
 new_weight = config.weight_evaluate(); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / delta_t_L_U * new_weight / config.weight_value;
}

dcomplex remove::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 config.insert(p, removed_pt);
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order; // order before removal
 quick_exit = (k - 2 < params->min_perturbation_order);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the operators for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // if remove p1, and p2 is later, ????? FIXME
 removed_pt1 = config.get_config(p1);
 removed_pt2 = config.get_config(p2);
 config.remove2(p1, p2);
 new_weight = config.weight_evaluate(); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(delta_t_L_U, 2) * new_weight / config.weight_value;
}

dcomplex remove2::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 config.insert2(p1, p2, removed_pt1, removed_pt2);
}

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                        // Choose one of the operators
 removed_pt = config.get_config(p); // old time, to be saved for the removal case

 auto new_pt = get_random_point(); // new time
 config.change_config(p, new_pt);
 new_weight = config.weight_evaluate();

 // The Metropolis ratio
 return new_weight / config.weight_value;
}

dcomplex shift::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 config.change_config(p, removed_pt);
}

// ------------ QMC additional time swap move --------------------------------------

dcomplex weight_swap::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif

 auto k = config.order;
 keldysh_contour_pt tau;

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                     // Choose one of the operators
 auto swap_pt = config.get_config(p); // point whose time is to be swapped
 tau = config.get_left_input();  // integrand left input point
 save_swap_pt = swap_pt;         // save for reject case
 save_tau = tau;                 // save for reject case
 // swap times:
 double tau_time = tau.t;
 tau.t = swap_pt.t;
 tau.k_index = rng(2); // random keldysh index
 swap_pt.t = tau_time;

 // replace points with swapped times
 config.change_config(p, swap_pt);
 config.change_left_input(tau);

 new_weight = config.weight_evaluate();

 // The Metropolis ratio
 return new_weight / config.weight_value;
}

dcomplex weight_swap::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void weight_swap::reject() {
 if (quick_exit) return;
 config.change_config(p, save_swap_pt);
 config.change_left_input(save_tau);
}

// ------------ QMC additional time shift move --------------------------------------

dcomplex weight_shift::attempt() {
#ifdef REGISTER_CONFIG
 config.register_config();
#endif
 // No quick exit for this move, all orders are concerned

 keldysh_contour_pt tau = config.get_left_input(); // integrand left input point
 save_tau = tau;                                   // save for reject case
 tau.t = rng(delta_t) - params->interaction_start; // random time
 tau.k_index = rng(2);                             // random keldysh index
 config.change_left_input(tau);

 new_weight = config.weight_evaluate();

 // The Metropolis ratio
 return new_weight / config.weight_value;
}

dcomplex weight_shift::accept() {
 config.weight_value = new_weight;
 config.accepted_kernels = config.current_kernels;
 return 1.0;
}

void weight_shift::reject() {
 // No quick exit for this move, all orders are concerned
 config.change_left_input(save_tau);
}
}
