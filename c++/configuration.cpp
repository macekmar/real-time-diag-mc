#include "./configuration.hpp"

//#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

Configuration::Configuration(g0_keldysh_alpha_t green_function, std::vector<keldysh_contour_pt> annihila_pts,
                             std::vector<keldysh_contour_pt> creation_pts, int max_order,
                             std::pair<double, double> singular_thresholds, bool kernels_comput,
                             bool nonfixed_op, int cycles_trapped_thresh)
   : singular_thresholds(singular_thresholds),
     cofactor_threshold(2. * singular_thresholds.first),
     order(0),
     potential(1.),
     max_order(max_order),
     kernels_comput(kernels_comput),
     nonfixed_op(nonfixed_op),
     spin_dvpt(creation_pts[0].s),
     cycles_trapped_thresh(cycles_trapped_thresh) {


 weight_sum = array<double, 1>(max_order + 1);
 weight_sum() = 0;
 nb_values = array<long, 1>(max_order + 1);
 nb_values() = 0;
 nb_cofact = array<long, 1>(max_order);
 nb_cofact() = 0;
 nb_inverse = array<long, 1>(max_order);
 nb_inverse() = 0;

 // Initialize the M-matrices. 100 is the initial alocated space.
 for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_n_operations_before_check(1000);

 matrices[0].set_singular_threshold(singular_thresholds.first);
 matrices[1].set_singular_threshold(singular_thresholds.second);

 if (annihila_pts.size() != creation_pts.size())
  TRIQS_RUNTIME_ERROR << "`annihila_pts` and `creation_pts` have different sizes";

 // inserting external Keldysh contour points
 for (size_t i = 0; i < creation_pts.size(); ++i) {
  if (annihila_pts[i].s != creation_pts[i].s) // both points assumed to have same spin
   TRIQS_RUNTIME_ERROR << "Pairs of annihilation and creation points must have the same spin";

  matrices[annihila_pts[i].s].insert_at_end(annihila_pts[i], creation_pts[i]);
 }

 current_kernels = array<dcomplex, 2>(max_order + matrices[spin_dvpt].size(), 2);
 current_kernels() = 0;
 accepted_kernels = array<dcomplex, 2>(max_order + matrices[spin_dvpt].size(), 2);
 accepted_kernels() = 0;

 // Initialize weight value
 evaluate();
 accept_config();
}

void Configuration::insert(vertex_t vtx) {
 auto pt = vtx.get_up_pt();
 matrices[up].insert(0, 0, pt, pt);
 pt = vtx.get_down_pt();
 matrices[down].insert(0, 0, pt, pt);

 potential_list.insert(0, vtx.potential);
 potential *= vtx.potential;

 order++;
};

// insert two vertices at once.
void Configuration::insert2(vertex_t vtx1, vertex_t vtx2) {
 auto pt1 = vtx1.get_up_pt();
 auto pt2 = vtx2.get_up_pt();
 matrices[up].insert2(0, 1, 0, 1, pt1, pt2, pt1, pt2);
 pt1 = vtx1.get_down_pt();
 pt2 = vtx2.get_down_pt();
 matrices[down].insert2(0, 1, 0, 1, pt1, pt2, pt1, pt2);

 potential_list.insert(0, vtx1.potential);
 potential_list.insert(1, vtx2.potential); // TODO: check this is the correct order
 potential *= vtx1.potential * vtx2.potential;

 order += 2;
};

void Configuration::remove(int k) {
 for (auto& m : matrices) m.remove(k, k);

 potential /= potential_list[k];
 potential_list.erase(k);

 order--;
};

/**
 * Remove the vertices at positions k1 and k2.
 * k1 and k2 must be two *distinct* existing positions.
 */
void Configuration::remove2(int k1, int k2) {
 for (auto& m : matrices) m.remove2(k1, k2, k1, k2);

 potential /= potential_list[k1] * potential_list[k2];
 potential_list.erase(k1, k2);

 order -= 2;
};

void Configuration::change_vertex(int k, vertex_t vtx) {
 auto pt = vtx.get_up_pt();
 matrices[up].change_row(k, pt);
 matrices[up].change_col(k, pt);
 pt = vtx.get_down_pt();
 matrices[down].change_row(k, pt);
 matrices[down].change_col(k, pt);

 potential *= vtx.potential / potential_list[k];
 potential_list[k] = vtx.potential;
};

vertex_t Configuration::get_vertex(int p) const {
 auto pt_up = matrices[up].get_x(p);
 auto pt_down = matrices[down].get_x(p);
 return {pt_up.x, pt_down.x, pt_up.t, pt_up.k_index, potential_list[p]}; // no consistency check is done
};

// -----------------------
 /* Evaluate the kernels for the current configuration. Fill `current_kernels` appropriately
  * and return the corresponding weight (as a real positive value):
  * W(\vec{u}) = \sum_{p=1}^n \sum_{a=0,1}| K_p^a(\vec{u}) |
  * If order n=0, `current_kernels` is not changed and the arbitrary weight 1 is returned.
  * */
 // TODO: write down the formula this implements
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

  if (matrix_dvpt.get_cond_nb() > cofactor_threshold or (not std::isnormal(std::abs(det_dvpt)))) {
   // matrix is singular, calculate cofactors
   nb_cofact(order - 1)++;

   if (nonfixed_op)
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
    if (nonfixed_op)
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

// -----------------------
 /* Evaluate the kernels and weight of the current configuration.
  * Store them into `current_kernels` and `current_weight`.
  */
void Configuration::evaluate() {
 dcomplex value;
 if (kernels_comput)
  value = kernels_evaluate();
 else
  value = potential * keldysh_sum();

 weight_sum(order) += std::abs(value);
 nb_values(order)++;
 current_weight = value;
}

// -----------------------
void Configuration::accept_config() {
 if (cycles_trapped > 100) std::cout << "Trapped " << cycles_trapped << " cycles" << std::endl;
 cycles_trapped = 0;
 accepted_weight = current_weight;
 accepted_kernels() = current_kernels();
}

// -----------------------
void Configuration::incr_cycles_trapped() {
 cycles_trapped++;
 if (cycles_trapped % cycles_trapped_thresh == 0) {
  evaluate();
  accepted_weight = current_weight;
  accepted_kernels() = current_kernels();
  // do not reset cycles_trapped to 0
 }
}

// -----------------------
// build configuration signature
// TODO: add orbitals in the signature
std::vector<double> Configuration::signature() {
 std::vector<double> signa;
 for (int i = 0; i < order; ++i) {
  signa.emplace_back(get_vertex(i).t);
 }
 return signa;
};

// -----------------------
// Register the configuration as if it has been accepted (accepted weight is stored).
// Increment multiplicity if it didn't change since last registration.
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

// -----------------------
// Register the configuration as if it has been attempted only (current weight is stored).
void Configuration::register_attempted_config() {
 auto config = signature();

 config_list.emplace_back(config);
 config_weight.emplace_back(current_weight);
};

// -----------------------
// TODO: add orbitals in the print
void Configuration::print() {
 std::cout << std::endl;
 int n;
 for (auto& m : matrices) {
  n = m.size();
  for (int i = 0; i < n; ++i) std::cout << "(" << m.get_x(i).t << ", " << m.get_x(i).k_index << "), ";
  std::cout << std::endl;
  for (int i = 0; i < n; ++i) std::cout << "(" << m.get_y(i).t << ", " << m.get_y(i).k_index << "), ";
  std::cout << std::endl;
  std::cout << std::endl;
 }
};
