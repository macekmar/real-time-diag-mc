#include "./moves.hpp"
#include <triqs/det_manip.hpp>

namespace moves {

std::vector<double> prepare_U(std::vector<double> U) {
 U.insert(U.begin(), 1.0); // so that U[k] is U at order k
 return U;
};

// /**
//  * Constructor for moves in the AuxMC Markov chain: set U to U_aux
//  */
template<>
common<ConfigurationAuxMC>::common(ConfigurationAuxMC &config, const solve_parameters_t &params,
  triqs::mc_tools::random_generator &rng, const RandomVertexGenerator &rvg) 
  : config(config), params(params), rng(rng), rvg(rvg), U(prepare_U(params.U)) {
 
 if (!params.U_aux.empty()) {
  U = prepare_U(params.U_aux);
 }
};

// ------------ QMC insertion move --------------------------------------
template <typename Conf>
dcomplex insert<Conf>::attempt() {
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

template <typename Conf>
dcomplex insert<Conf>::accept() {
 config.accept_config();
 return 1.0;
}

template <typename Conf>
void insert<Conf>::reject() {
 if (quick_exit) return;
 config.remove(0);
}

// ------------ QMC double-insertion move --------------------------------------

template <typename Conf>
dcomplex insert2<Conf>::attempt() {
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

template <typename Conf>
dcomplex insert2<Conf>::accept() {
 config.accept_config();
 return 1.0;
}

template <typename Conf>
void insert2<Conf>::reject() {
 if (quick_exit) return;
 auto k = config.order;
 config.remove2(0, 1);
}

//// ------------ QMC removal move --------------------------------------

template <typename Conf>
dcomplex remove<Conf>::attempt() {
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

template <typename Conf>
dcomplex remove<Conf>::accept() {
 config.accept_config();
 return 1.0;
}

template <typename Conf>
void remove<Conf>::reject() {
 if (quick_exit) return;
 config.insert(p, removed_vtx); // position of vertex is irrelevant
}

// ------------ QMC double-removal move --------------------------------------

template <typename Conf>
dcomplex remove2<Conf>::attempt() {
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

template <typename Conf>
dcomplex remove2<Conf>::accept() {
 config.accept_config();
 return 1.0;
}

template <typename Conf>
void remove2<Conf>::reject() {
 if (quick_exit) return;
 config.insert2(p1, p2, removed_vtx1, removed_vtx2); // position of vertex is irrelevant
}

// ------------ QMC vertex shift move --------------------------------------

template <typename Conf>
dcomplex shift<Conf>::attempt() {
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

template <typename Conf>
dcomplex shift<Conf>::accept() {
 config.accept_config();
 return 1.0;
}

template <typename Conf>
void shift<Conf>::reject() {
 if (quick_exit) return;
 config.change_vertex(p, removed_vtx);
}

// ------------ QMC Auxillary MC move --------------------------------------

dcomplex auxmc::attempt() {
 before_attempt();

 int i;
 // Set aux_config state to main config state
 // If the previous move was accepted, we do not have to reset aux_config
 auto k_current = config.order;
 if (!move_accepted) {
  vertices = config.vertices_list();
  aux_config->reset_to_vertices(vertices);
  aux_config->evaluate();
  aux_config->accept_config();
 }
 auto aux_accepted_weight = aux_config->accepted_weight;
 // Also save the current config
 old_vertices = vertices;
 // Do auxiliary MC run
 // TODO: in `solver_core.cpp` in `clock_callback` the argument is the variable
 //       `max_time instead` of -1 (which is the default value of `max_time`)
 aux_mc->run(params.nb_aux_mc_cycles, 1, triqs::utility::clock_callback(-1), false);
 auto aux_current_weight = aux_config->accepted_weight;
 // Set the main config state to final aux_config state
 auto k_attempted = aux_config->order;
 vertices = aux_config->vertices_list();
 config.reset_to_vertices(vertices);
 config.evaluate();
 after_attempt();

 double U_prod = 1.0;
 double U_prod_aux = 1.0;
 /**
  * Marjan: 
  * aux_..._weights do not include U_aux although it is used in the aux Markov
  * chain.
  * U_qmc is just an additional parameter to overcome different magnitudes of
  * qmc weights.
 */
 int sgn = k_attempted > k_current ? 1 : -1;
 for (i = k_current + sgn; i != k_attempted + sgn; i += sgn) {
  U_prod *= U[i];
  U_prod_aux *= U_aux[i];
 };
 if (sgn < 0) U_prod = 1.0/U_prod;
 if (sgn < 0) U_prod_aux = 1.0/U_prod_aux;


 // The Metropolis ratio;
 return U_prod/U_prod_aux * aux_accepted_weight/aux_current_weight * config.current_weight / config.accepted_weight;
}

dcomplex auxmc::accept() {
 config.accept_config();
 move_accepted = true;
 return 1.0;
}

void auxmc::reject() {
 if (quick_exit) return;
 config.reset_to_vertices(old_vertices);
 move_accepted = false;
}


template class insert<ConfigurationQMC>;
template class insert2<ConfigurationQMC>;
template class remove<ConfigurationQMC>;
template class remove2<ConfigurationQMC>;
template class shift<ConfigurationQMC>;

template class insert<ConfigurationAuxMC>;
template class insert2<ConfigurationAuxMC>;
template class remove<ConfigurationAuxMC>;
template class remove2<ConfigurationAuxMC>;
template class shift<ConfigurationAuxMC>;
}
