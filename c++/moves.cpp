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
 config.evaluate();

 // The Metropolis ratio;
 return delta_t_L_U / (k + 1) * config.current_weight / config.accepted_weight;
}

dcomplex insert::accept() {
 config.accept_config();
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

 config.evaluate();

 // The Metropolis ratio
 return delta_t_L_U * delta_t_L_U / ((k + 1) * (k + 2)) * config.current_weight / config.accepted_weight;
}

dcomplex insert2::accept() {
 config.accept_config();
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
 config.evaluate(); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k / delta_t_L_U * config.current_weight / config.accepted_weight;
}

dcomplex remove::accept() {
 config.accept_config();
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
 config.evaluate(); // recompute sum over keldysh indices

 // The Metropolis ratio
 return k * (k - 1) / pow(delta_t_L_U, 2) * config.current_weight / config.accepted_weight;
}

dcomplex remove2::accept() {
 config.accept_config();
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
 config.evaluate();

 // The Metropolis ratio
 return config.current_weight / config.accepted_weight;
}

dcomplex shift::accept() {
 config.accept_config();
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 config.change_config(p, removed_pt);
}

}
