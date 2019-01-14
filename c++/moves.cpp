#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace moves {

std::vector<double> prepare_U(std::vector<double> U) {
 U.insert(U.begin(), 1.0); // so that U[k] is U at order k
 return U;
};


// ------------ QMC insertion move --------------------------------------

dcomplex insert::attempt() {
 before_attempt();

 auto k = config.order; // order before adding a vertex
 quick_exit = is_quick_exit(k+1);
 if (quick_exit) return 0;

 // insert the new line and col.
 auto vtx = rvg(); // vertex random generator
 double proba = rvg.probability(vtx);
 config.insert(0, vtx);
 config.evaluate();

 after_attempt();

 // The Metropolis ratio;
 return U[k+1] / (proba * (k + 1)) * config.current_weight / config.accepted_weight;
}

dcomplex insert::accept() {
 config.accept_config();
 return 1.0;
}

void insert::reject() {
 if (quick_exit) return;
 config.remove(0);
}

// ------------ QMC double-insertion move --------------------------------------

dcomplex insert2::attempt() {
 before_attempt();

 auto k = config.order; // order before adding two vertices
 quick_exit = is_quick_exit(k+2);
 if (quick_exit) return 0;

 // insert the new lines and cols.
 auto vtx1 = rvg();
 auto vtx2 = rvg();
 double proba = rvg.probability(vtx1) * rvg.probability(vtx2);
 config.insert2(0, 1, vtx1, vtx2);

 config.evaluate();

 after_attempt();

 // The Metropolis ratio
 return U[k+2] * U[k+1] / (proba * (k + 1) * (k + 2)) * config.current_weight / config.accepted_weight;
}

dcomplex insert2::accept() {
 config.accept_config();
 return 1.0;
}

void insert2::reject() {
 if (quick_exit) return;
 auto k = config.order;
 config.remove2(0, 1);
}

//// ------------ QMC removal move --------------------------------------

dcomplex remove::attempt() {
 before_attempt();

 auto k = config.order; // order before removal
 quick_exit = is_quick_exit(k-1);
 if (quick_exit) return 0;

 // remove the line/col
 p = rng(k);                            // Choose one of the vertices for removal
 removed_vtx = config.get_vertex(p);     // store the vertex to be removed for later reject
 config.remove(p);                      // remove the vertex for all matrices
 config.evaluate(); // recompute sum over keldysh indices

 after_attempt();

 // The Metropolis ratio
 return rvg.probability(removed_vtx) * k / U[k] * config.current_weight / config.accepted_weight;
}

dcomplex remove::accept() {
 config.accept_config();
 return 1.0;
}

void remove::reject() {
 if (quick_exit) return;
 config.insert(p, removed_vtx); // position of vertex is irrelevant
}

// ------------ QMC double-removal move --------------------------------------

dcomplex remove2::attempt() {
 before_attempt();

 auto k = config.order; // order before removal
 quick_exit = is_quick_exit(k-2);
 if (quick_exit) return 0;

 // remove the lines/cols
 p1 = rng(k);        // Choose one of the vertices for removal
 p2 = rng(k - 1);    //
 if (p2 >= p1) p2++; // p1 and p2 must be distinct
 removed_vtx1 = config.get_vertex(p1);
 removed_vtx2 = config.get_vertex(p2);
 config.remove2(p1, p2);
 config.evaluate(); // recompute sum over keldysh indices

 after_attempt();

 // The Metropolis ratio
 return rvg.probability(removed_vtx1) * rvg.probability(removed_vtx2) * k * (k - 1) / (U[k] * U[k-1]) * config.current_weight / config.accepted_weight;
}

dcomplex remove2::accept() {
 config.accept_config();
 return 1.0;
}

void remove2::reject() {
 if (quick_exit) return;
 config.insert2(p1, p2, removed_vtx1, removed_vtx2); // position of vertex is irrelevant
}

// ------------ QMC vertex shift move --------------------------------------

dcomplex shift::attempt() {
 before_attempt();

 auto k = config.order; // order

 quick_exit = (k == 0); // In particular if k = 0
 if (quick_exit) return 0;
 p = rng(k);                        // Choose one of the vertices
 removed_vtx = config.get_vertex(p); // old vertex, to be saved for the removal case

 auto new_vtx = rvg(); // new vertex
 double proba = rvg.probability(new_vtx);
 config.change_vertex(p, new_vtx);
 config.evaluate();

 after_attempt();

 // The Metropolis ratio
 return rvg.probability(removed_vtx) / proba * config.current_weight / config.accepted_weight;
}

dcomplex shift::accept() {
 config.accept_config();
 return 1.0;
}

void shift::reject() {
 if (quick_exit) return;
 config.change_vertex(p, removed_vtx);
}

// ------------ QMC Auxillary MC move --------------------------------------

dcomplex auxmc::attempt() {
 before_attempt();

 int i;
//std::cout << mc->get_current_cycle_number() << std::endl; 
 // Set aux_config state to main config state
 auto k_current = config.order;
 vertices = config.vertices_list();
 aux_config->reset_to_vertices(vertices);
 aux_config->evaluate();
 auto aux_accepted_weight = aux_config->current_weight;
 // Also save the current config
 old_vertices = vertices;

 // Do auxiliary MC run
 // TODO: in solver_core.cpp as clock_callback argument is 
 //       max_time instead of -1 (which is the default value of max_time)
 aux_mc->run(params.aux_nb_cycles, 1, triqs::utility::clock_callback(-1), false);
 
 auto aux_current_weight = aux_config->accepted_weight;

 // Set main config state to final aux_config state
 auto k_attempted = aux_config->order;
 vertices = aux_config->vertices_list();
 config.reset_to_vertices(vertices);
 config.evaluate();
 after_attempt();

 double U_prod = 1;
 int sgn = k_attempted > k_current ? 1 : -1;
 for (i = 0; i < abs(k_attempted - k_current); i++) {
  U_prod *= U[k_current + sgn*i];
 };

 // The Metropolis ratio;
 U_prod = 1.0;
 //std::cout << aux_accepted_weight << aux_current_weight << config.current_weight << config.accepted_weight << std::endl;
 return U_prod * aux_accepted_weight/aux_current_weight * config.current_weight / config.accepted_weight;
}

dcomplex auxmc::accept() {
 config.accept_config();
 return 1.0;
}

void auxmc::reject() {
 if (quick_exit) return;
 config.reset_to_vertices(old_vertices);
}

}
