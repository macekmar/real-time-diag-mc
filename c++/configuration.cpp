#include "./configuration.hpp"
#include <algorithm>


//#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

/**
 * The Configuration class represents the current state of the Markov chain.
 * Its instance should be unique.
 *
 * Vertices are added, removed or changed by the Monte-Carlo moves. The
 * Configuration object allows to evaluate the weight and kernels of a
 * configuration and to keep memory of the previously accepted weight and
 * kernels.
 */
Configuration::Configuration(g0_keldysh_alpha_t green_function, const solve_parameters_t &params)
   : params(params),
     cofactor_threshold(2. * params.singular_thresholds.first),
     order(0),
     potential(1.) {

 if (not params.cycles_trapped_thresh > 0)
  TRIQS_RUNTIME_ERROR << "cycles_trapped_treshold must be > 0.";


 set_ops();
 spin_dvpt = creation_pts.front().s;

 nb_cofact = array<long, 1>(params.max_perturbation_order);
 nb_cofact() = 0;
 nb_inverse = array<long, 1>(params.max_perturbation_order);
 nb_inverse() = 0;

 // Initialize the M-matrices. 100 is the initial alocated space.
 for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_n_operations_before_check(1000);

 matrices[0].set_singular_threshold(params.singular_thresholds.first);
 matrices[1].set_singular_threshold(params.singular_thresholds.second);

 set_default_values();


 current_kernels = array<dcomplex, 2>(params.max_perturbation_order + matrices[spin_dvpt].size(), 2);
 current_kernels() = 0;
 accepted_kernels = array<dcomplex, 2>(params.max_perturbation_order + matrices[spin_dvpt].size(), 2);
 accepted_kernels() = 0;
/* This is now in each child class
 * // Initialize weight value
 * evaluate();
 * accept_config();
 */
}

/**
 * Set creation and annihilation contour points from parameters
 * 
 * Marjan: Previously *_pts were defined only inside the constructor, but I
 *         need to access them every time I reset the configuration.
 * TODO: Of course, they don't change after the initialization????
 */
void Configuration::set_ops() {
 for (int rank = 0; rank < params.creation_ops.size(); ++rank) {
  creation_pts.push_back(make_keldysh_contour_pt(params.creation_ops[rank], rank));
  annihila_pts.push_back(make_keldysh_contour_pt(params.annihilation_ops[rank], rank));
 }

 if (annihila_pts.size() != creation_pts.size())
  TRIQS_RUNTIME_ERROR << "`annihila_pts` and `creation_pts` have different sizes";

 // inserting external Keldysh contour points
 std::list<keldysh_contour_pt>::iterator apt, cpt;
 for (cpt = creation_pts.begin(), apt = annihila_pts.begin(); 
      cpt != creation_pts.end() && apt != annihila_pts.end(); 
      ++cpt, ++apt) {
  if (apt->s != cpt->s) // both points assumed to have same spin
   TRIQS_RUNTIME_ERROR << "Pairs of annihilation and creation points must have the same spin";
 }
}

void Configuration::set_default_values() {
 std::list<keldysh_contour_pt>::iterator apt, cpt;
 for (cpt = creation_pts.begin(), apt = annihila_pts.begin(); 
      cpt != creation_pts.end() && apt != annihila_pts.end(); 
      ++cpt, ++apt) {
  matrices[apt->s].insert_at_end(*apt, *cpt);
  times_list_.insert(apt->t);
  times_list_.insert(cpt->t);
  orbitals_list_.insert(0, apt->x);
  orbitals_list_.insert(0, cpt->x);
}
}

/**
 * Insert a vertex at position k (before index k).
 */
void Configuration::insert(int k, vertex_t vtx) {
 times_list_.insert(vtx.t);
 orbitals_list_.insert(k, vtx.x_up);

 auto pt = vtx.get_up_pt();
 matrices[up].insert(k, k, pt, pt);
 pt = vtx.get_down_pt();
 matrices[down].insert(k, k, pt, pt);

 potential_list.insert(k, vtx.potential);
 potential *= vtx.potential;

 order++;
};

/**
 * Insert two vertices such that `vtx1` is now at position k1 and `vtx2` at position k2.
 */
void Configuration::insert2(int k1, int k2, vertex_t vtx1, vertex_t vtx2) {
 times_list_.insert(vtx1.t);
 times_list_.insert(vtx2.t);
 orbitals_list_.insert2(k1, k2, vtx1.x_up, vtx2.x_up);

 auto pt1 = vtx1.get_up_pt();
 auto pt2 = vtx2.get_up_pt();
 matrices[up].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 pt1 = vtx1.get_down_pt();
 pt2 = vtx2.get_down_pt();
 matrices[down].insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);

 potential_list.insert2(k1, k2, vtx1.potential, vtx2.potential);
 potential *= vtx1.potential * vtx2.potential;

 order += 2;
};

/**
 * Remove the vertex at position `k`.
 */
void Configuration::remove(int k) {
 times_list_.erase(get_time(k)); // before matrix removal
 orbitals_list_.erase(k);

 for (auto& m : matrices) m.remove(k, k);

 potential /= potential_list[k];
 potential_list.erase(k);

 order--;
};

/**
 * Remove the vertices at positions `k1` and `k2`.
 */
void Configuration::remove2(int k1, int k2) {
 times_list_.erase(get_time(k1));
 times_list_.erase(get_time(k2));
 orbitals_list_.erase2(k1, k2);

 for (auto& m : matrices) m.remove2(k1, k2, k1, k2);

 potential /= potential_list[k1] * potential_list[k2];
 potential_list.erase2(k1, k2);

 order -= 2;
};

/**
 * Change vertex at position `k` into `vtx`.
 */
void Configuration::change_vertex(int k, vertex_t vtx) {
 times_list_.erase(get_time(k));
 times_list_.insert(vtx.t);
 orbitals_list_[k] = vtx.x_up;

 auto pt = vtx.get_up_pt();
 matrices[up].change_row(k, pt);
 matrices[up].change_col(k, pt);
 pt = vtx.get_down_pt();
 matrices[down].change_row(k, pt);
 matrices[down].change_col(k, pt);

 potential *= vtx.potential / potential_list[k];
 potential_list[k] = vtx.potential;
};

/**
 * Return a copy of the vertex at position `k`.
 *
 * Vertices are not actually stored, so it is reconstructed from the Keldysh
 * points in the matrices.
 */
vertex_t Configuration::get_vertex(int k) const {
 auto pt_up = matrices[up].get_x(k);
 auto pt_down = matrices[down].get_x(k);
 return {pt_up.x, pt_down.x, pt_up.t, pt_up.k_index, potential_list[k]}; // no consistency check is done
};

/**
 * Insert vertices consequently
 */
void Configuration::insert_vertices(std::list<vertex_t> vertices) {
 int i = 0;
 int begin_order = order;
 while (!vertices.empty()){
  insert(begin_order + i, vertices.front());
  vertices.pop_front();
  ++i;
 }
};


void Configuration::reset_to_vertices(std::list<vertex_t> vertices) {
 // Instead of 
 // remove_all();
 // which removes vertices one after another and recalculates the determinant
 // each time, we can clear all values and set back the default ones.

 times_list_.clear();
 orbitals_list_.clear(); 
 for (auto& m : matrices) m.clear();
 potential_list.clear();
 order = 0;
 potential = 1.0;

 set_default_values();

 insert_vertices(vertices);
};

inline timec_t Configuration::get_time(int k) const {
 return matrices[up].get_x(k).t;
};

/**
 * Returns a list of all vertices in the configuration
 */
 
std::list<vertex_t> Configuration::vertices_list() {
 std::list<vertex_t> vertices;
 for (int i = 0; i < order; i++){
  //std::cout << "works" << i << std::endl;
  vertices.push_back(get_vertex(i));
 };
 return vertices;
};

/**
 * Evaluate the kernels for the current configuration.
 *
 * Fill `current_kernels` appropriately and return the corresponding weight
 * (as a real positive value):
 * W(\vec{u}) = \sum_{p=1}^n \sum_{a=0,1}| K_p^a(\vec{u}) |
 * If order n=0, `current_kernels` is not changed and the arbitrary weight 1
 * is returned.
 *
 * TODO: write down the formula this implements
 */
double Configuration::kernels_evaluate() {
 if (order == 0) return 1.; // is this a good value ?
 if (order > 63) TRIQS_RUNTIME_ERROR << "order overflow";

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[0].regenerate();
 matrices[1].regenerate();
#endif

 current_kernels() = 0;
 dcomplex det_fix, det_dvpt, inv_value;
 size_t k_index = 0;
 auto& matrix_dvpt = matrices[spin_dvpt];
 auto& matrix_fix = matrices[1 - spin_dvpt];
 array<dcomplex, 1> cofactors(matrix_dvpt.size());

 keldysh_contour_pt pt;
 int sign = -1; // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << order; // shifts the bits order times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : order) -
            1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
               // to 1. ~n has bites inversed compared with n.

  for (auto spin : {up, down}) {
   pt = flip_index(matrices[spin].get_x(nlc));
   matrices[spin].change_row(nlc, pt);
   matrices[spin].change_col(nlc, pt);
  }

  det_fix = sign * matrix_fix.determinant();
  det_dvpt = matrix_dvpt.determinant();

  if (matrix_dvpt.get_cond_nb() > cofactor_threshold or (not std::isnormal(std::abs(det_dvpt))) or params.method == 2) {
   // matrix is singular, calculate cofactors
   nb_cofact(order - 1)++;

   if (params.nonfixed_op)
    cofactors = cofactor_col(matrix_dvpt, order, matrix_dvpt.size());
   else
    cofactors = cofactor_row(matrix_dvpt, order, matrix_dvpt.size());

   for (size_t p = 0; p < matrix_dvpt.size(); ++p) {
    k_index = matrix_dvpt.get_x(p).k_index;
    current_kernels(p, k_index) += cofactors(p) * det_fix;
   }
  } else {
   nb_inverse(order - 1)++;

   for (size_t p = 0; p < matrix_dvpt.size(); ++p) {
    if (params.nonfixed_op)
     inv_value = matrix_dvpt.inverse_matrix(order, p);
    else
     inv_value = matrix_dvpt.inverse_matrix(p, order);

    k_index = matrix_dvpt.get_x(p).k_index;
    current_kernels(p, k_index) += inv_value * det_dvpt * det_fix;
   }
  }
  sign = -sign;
 }

 current_kernels() *= potential;

 return sum(abs(current_kernels()));
};

/**
 * Accept the current configuration.
 *
 * This simply copy `current_weight` and `current_kernels` into
 * `accepted_weight` and `accepted_kernels`.
 */
void Configuration::accept_config() {
 //if (cycles_trapped > 100) std::cout << "Trapped " << cycles_trapped << " cycles" << std::endl;
 cycles_trapped = 0;
 accepted_weight = current_weight;
 accepted_kernels() = current_kernels();
}

/**
 * Increment the trapped counter, and if threshold has been reached, reevaluate
 * the current configuration.
 *
 * The weight and kernels are automatically accepted, so this method should be
 * called on accepted configurations only.
 */
void ConfigurationQMC::incr_cycles_trapped() {
 cycles_trapped++;
 if (cycles_trapped % params.cycles_trapped_thresh == 0) {
  evaluate();
  accepted_weight = current_weight;
  accepted_kernels() = current_kernels();
  // do not reset cycles_trapped to 0, its done after acceptation
 }
}
void ConfigurationAuxMC::incr_cycles_trapped() {
 cycles_trapped++;
 if (cycles_trapped % params.cycles_trapped_thresh == 0) {
  evaluate();
  accepted_weight = current_weight;
  accepted_kernels() = current_kernels();
  // do not reset cycles_trapped to 0, its done after acceptation
 }
}


/**
 * Build configuration signature.
 *
 * This is used to store the configurations in a file. It is not supposed to
 * give exhaustive information.
 * TODO: add orbitals in the signature
 */
std::vector<double> Configuration::signature() {
 std::vector<double> signa;
 for (int i = 0; i < order; ++i) {
  signa.emplace_back(get_vertex(i).t);
 }
 return signa;
};

/**
 * Register the configuration as if it has been accepted (accepted weight is stored).
 * Increment multiplicity if it didn't change since last registration.
 */
void Configuration::register_accepted_config() {
 auto config = signature();

 if (config_list.size() > 0 && config == config_list[config_list.size() - 1]) // short-circuit eval
  config_mult[config_mult.size() - 1]++;
 else {
  config_list.emplace_back(config);
  config_mult.emplace_back(1);
  config_weight.emplace_back(accepted_weight);
 }
};

/**
 * Register the configuration as if it has been attempted only (current weight is stored).
 * No use of multiplicity here.
 */
void Configuration::register_attempted_config() {
 auto config = signature();

 config_list.emplace_back(config);
 config_weight.emplace_back(current_weight);
};

/**
 * Print the current configuration.
 *
 * Prints the lists of contour points which composes the matrices, as well as
 * the list of potentials of vertices.
 */
void Configuration::print() {
 std::cout << std::endl;
 int n;
 for (auto& m : matrices) {
  n = m.size();
  std::cout << "x = ";
  for (int i = 0; i < n; ++i) std::cout << "(" << m.get_x(i).x << ", " << m.get_x(i).s << ", " << m.get_x(i).t << ", " << m.get_x(i).k_index << "), ";
  std::cout << std::endl;
  std::cout << "y = ";
  for (int i = 0; i < n; ++i) std::cout << "(" << m.get_y(i).x << ", " << m.get_y(i).s << ", " << m.get_y(i).t << ", " << m.get_y(i).k_index << "), ";
  std::cout << std::endl;
 }
 std::cout << "V = ";
 for (auto it = potential_list.begin(); it != potential_list.end(); ++it) {
  std::cout << *it << ", ";
 }
 std::cout << std::endl;
 std::cout << "V product = " << potential;
 std::cout << std::endl;
 std::cout << std::endl;
};

// ------------ QMC ------------ ----------------------------------------------
void ConfigurationQMC::evaluate() {
 if (params.method != 0)
  current_weight = kernels_evaluate();
 else
  // no absolute value as this is used for sn
  current_weight = potential * keldysh_sum();
 }


// ------------ Auxillary MC --------------------------------------------------
dcomplex ConfigurationAuxMC::_eval(std::vector<std::tuple<orbital_t, orbital_t, timec_t>> vertices) {
 if (vertices.size() > params.max_perturbation_order)
  TRIQS_RUNTIME_ERROR << "Too many vertices compared to the max perturbation order.";

 auto pot_data = make_potential_data(params.nb_orbitals, params.potential);

 config_qmc.remove_all();

 orbital_t i, j;
 double pot;
 for (auto it = vertices.rbegin(); it != vertices.rend(); ++it) {
  i = std::get<0>(*it);
  j = std::get<1>(*it);
  pot = pot_data.potential_of(i, j);

  if (i < 0 or i >= params.nb_orbitals or j < 0 or j >= params.nb_orbitals)
  TRIQS_RUNTIME_ERROR << "Orbital not recognized";
  if (pot == 0.)
  TRIQS_RUNTIME_ERROR << "These orbitals have no potential"; // Configuration should never be fed with a zero potential vertex, or it breaks down.

  config_qmc.insert(0, {i, j, std::get<2>(*it), 0, pot});
 }

 config_qmc.evaluate();
 config_qmc.remove_all();

 return config_qmc.current_weight; //Marjan:TODO: in abs()?
};

void ConfigurationAuxMC::evaluate() {
    
 dcomplex prod;
 std::list<vertex_t>::iterator vtx;    
 std::list<vertex_t> vertices = vertices_list();

 timec_t t1, t2;

 if (vertices.size() == 0) { current_weight = 1.0;}
 else {
  vertices.sort(compare_vertices);

  prod = _eval({std::make_tuple(0, 0, vertices.begin()->t)});
  for (vtx = vertices.begin(); vtx != vertices.end(); ++vtx) {
   if (std::next(vtx, 1) == vertices.end()) break;
   if (vtx != vertices.end()) {
     t1 = vtx->t;
     t2 = std::next(vtx, 1)->t;
     prod *= _eval({std::make_tuple(0, 0, t2 - t1)});
    }
  }
  current_weight = prod;
 }
}



