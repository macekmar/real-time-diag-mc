#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace moves {

// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {
 if (params->store_configurations == 1) config.register_accepted_config();

 auto k = config.order; // order before adding a vertex
 quick_exit = is_quick_exit(k+1);
 if (quick_exit) return 0;

 // insert the new line and col.
 auto vtx = get_random_vertex();
 config.insert(k, vtx);
 config.evaluate();

 if (params->store_configurations == 2) config.register_attempted_config();

 // The Metropolis ratio;
 return normalization / (k + 1) * config.current_weight / config.accepted_weight;
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
 if (params->store_configurations == 1) config.register_accepted_config();

 auto k = config.order; // order before adding two vertices
 quick_exit = is_quick_exit(k+2);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto vtx1 = get_random_vertex();
 auto vtx2 = get_random_vertex();
 config.insert2(k, k + 1, vtx1, vtx2);

 config.evaluate();

 if (params->store_configurations == 2) config.register_attempted_config();

 // The Metropolis ratio
 return normalization * normalization / ((k + 1) * (k + 2)) * config.current_weight / config.accepted_weight;
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
 if (params->store_configurations == 1) config.register_accepted_config();

 auto k = config.order; // order before removal
 quick_exit = is_quick_exit(k-1);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                            // Choose one of the vertices for removal
 removed_vtx = config.get_vertex(p);     // store the vertex to be removed for later reject
 config.remove(p);                      // remove the vertex for all matrices
 config.evaluate(); // recompute sum over keldysh indices

 if (params->store_configurations == 2) config.register_attempted_config();

 // The Metropolis ratio
 return k / normalization * config.current_weight / config.accepted_weight;
}

dcomplex remove::accept() {
 config.accept_config();
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 config.insert(p, removed_vtx);
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {
 if (params->store_configurations == 1) config.register_accepted_config();

 auto k = config.order; // order before removal
 quick_exit = is_quick_exit(k-2);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the vertices for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // if remove p1, and p2 is later, ????? FIXME
 removed_vtx1 = config.get_vertex(p1);
 removed_vtx2 = config.get_vertex(p2);
 config.remove2(p1, p2);
 config.evaluate(); // recompute sum over keldysh indices

 if (params->store_configurations == 2) config.register_attempted_config();

 // The Metropolis ratio
 return k * (k - 1) / pow(normalization, 2) * config.current_weight / config.accepted_weight;
}

dcomplex remove2::accept() {
 config.accept_config();
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 config.insert2(p1, p2, removed_vtx1, removed_vtx2);
}

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {
 if (params->store_configurations == 1) config.register_accepted_config();

 auto k = config.order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                        // Choose one of the vertices
 removed_vtx = config.get_vertex(p); // old vertex, to be saved for the removal case

 auto new_vtx = get_random_vertex(); // new vertex
 config.change_vertex(p, new_vtx);
 config.evaluate();

 if (params->store_configurations == 2) config.register_attempted_config();

 // The Metropolis ratio
 return config.current_weight / config.accepted_weight;
}

dcomplex shift::accept() {
 config.accept_config();
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 config.change_vertex(p, removed_vtx);
}

}
