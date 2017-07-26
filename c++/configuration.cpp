#include "./configuration.hpp"

//#define REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE

Configuration::Configuration(g0_keldysh_alpha_t green_function, std::vector<keldysh_contour_pt> annihila_pts,
                             std::vector<keldysh_contour_pt> creation_pts, int max_order,
                             std::pair<double, double> singular_thresholds, bool kernels_comput = true)
   : singular_thresholds(singular_thresholds),
     order(0),
     max_order(max_order),
     kernels_comput(kernels_comput) {


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
 for (auto spin : {up, down}) matrices[spin].set_n_operations_before_check(100);

 matrices[0].set_singular_threshold(singular_thresholds.first);
 matrices[1].set_singular_threshold(singular_thresholds.second);

 // inserting first annihilation point
 if (annihila_pts[0].x != 0 or
     creation_pts[0].x != 0) // spin of creation_pts[0] is assumed same as annihila_pts[0]
  TRIQS_RUNTIME_ERROR << "First points must have spin 0";
 matrices[0].insert_at_end(annihila_pts[0], creation_pts[0]);
 crea_k_ind.push_back(creation_pts[0].k_index);
 // inserting other points
 for (size_t i = 1; i < creation_pts.size(); ++i) {
  if (annihila_pts[i].x != creation_pts[i].x) // both points assumed to have same spin
   TRIQS_RUNTIME_ERROR << "Pairs of annihilation and creation points must have the same spin";
  size_t mat_ind = annihila_pts[i].x == 0 ? 0 : 1;
  annihila_pts[i].x = 0;
  creation_pts[i].x = 0;
  matrices[mat_ind].insert_at_end(annihila_pts[i], creation_pts[i]);
  if (mat_ind == 0) crea_k_ind.push_back(creation_pts[i].k_index);
 }

 current_kernels = array<dcomplex, 2>(max_order + matrices[0].size(), 2);
 current_kernels() = 0;
 accepted_kernels = array<dcomplex, 2>(max_order + matrices[0].size(), 2);
 accepted_kernels() = 0;

 // Initialize weight value
 evaluate();
 accept_config();
}

void Configuration::insert(int k, keldysh_contour_pt pt) {
 for (auto& m : matrices) m.insert(k, k, pt, pt);
 order++;
};

void Configuration::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto& m : matrices) m.insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
 order += 2;
};

void Configuration::remove(int k) {
 for (auto& m : matrices) m.remove(k, k);
 order--;
};

void Configuration::remove2(int k1, int k2) {
 for (auto& m : matrices) m.remove2(k1, k2, k1, k2);
 order -= 2;
};

void Configuration::change_config(int k, keldysh_contour_pt pt) {
 // for (auto& m : matrices) m.change_one_row_and_one_col(k, k, pt, pt);
 for (auto& m : matrices) {
  m.change_row(k, pt);
  m.change_col(k, pt);
 };
};

keldysh_contour_pt Configuration::get_config(int p) const {
 return matrices[0].get_x(p);
}; // assuming alpha is a single point

// -----------------------
double Configuration::kernels_evaluate() {
 /* Evaluate the kernels for the current configuration. Fill `current_kernels` appropriately
  * and return the corresponding weight (as a real positive value):
  * W(\vec{u}) = \sum_{p=0}^n \sum_{a=0,1}| K_p^a(\vec{u}) |
  * If order n=0, `current_kernels` is not changed and the arbitrary weight 1 is returned.
  * */
 // TODO: write down the formula this implements
 if (order == 0) return 1.; // is this a good value ?
 if (order > 63) TRIQS_RUNTIME_ERROR << "order overflow";

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[0].regenerate();
 matrices[1].regenerate();
#endif

 current_kernels() = 0;
 dcomplex det1, det0, inv_value;
 size_t ap[64] = {0};
 size_t k_index = 0;

 keldysh_contour_pt pt;
 int sign = -1; // Starting with a flip, so -1 -> 1, which is needed in the first iteration
 auto two_to_k = uint64_t(1) << order; // shifts the bits order times to the left
 for (uint64_t n = 0; n < two_to_k; ++n) {

  // The bit to flip to obtain the next element. Will be the index of line/col to be changed.
  // Cf Numerical Recipes, sect. 20.2
  int nlc = (n < two_to_k - 1 ? ffs(~n) : order) -
            1; // ffs starts at 1, returns the position of the 1st (least significant) bit set
               // to 1. ~n has bites inversed compared with n.
  ap[nlc] = 1 - ap[nlc];

  for (auto spin : {up, down}) {
   pt = flip_index(matrices[spin].get_x(nlc));
   matrices[spin].change_row(nlc, pt);
   matrices[spin].change_col(nlc, pt);
  }

  det1 = sign * matrices[1].determinant();
  det0 = matrices[0].determinant();

  if (not std::isnormal(std::abs(det0))) {
   // matrix is singular, calculate cofactors
   nb_cofact(order - 1)++;
   auto cofactors = cofactor_row(matrices[0], order, matrices[0].size());
   for (size_t p = 0; p < matrices[0].size(); ++p) {
    k_index = p < order ? ap[p] : crea_k_ind[p - order];
    current_kernels(p, k_index) += cofactors(p) * det1;
   }
  } else {
   nb_inverse(order - 1)++;
   for (size_t p = 0; p < matrices[0].size(); ++p) {
    inv_value = matrices[0].inverse_matrix(p, order);
    k_index = p < order ? ap[p] : crea_k_ind[p - order];
    current_kernels(p, k_index) += inv_value * det0 * det1;
   }
  }
  sign = -sign;
 }

 return sum(abs(current_kernels()));
};

// -----------------------
void Configuration::evaluate() {
 /* Evaluate the kernels and weight of the current configuration.
  * Store them into `current_kernels` and `current_weight`.
  */
 dcomplex value;
 if (kernels_comput)
  value = kernels_evaluate();
 else
  value = keldysh_sum();

 weight_sum(order) += std::abs(value);
 nb_values(order)++;
 current_weight = value;
}

// -----------------------
void Configuration::accept_config() {
 accepted_weight = current_weight;
 accepted_kernels() = current_kernels();
}

// -----------------------
void Configuration::register_config() {
 if (stop_register) return;

 int threshold = 1e7 / triqs::mpi::communicator().size();
 if (config_list.size() > threshold) {
  if (triqs::mpi::communicator().rank() == 0)
   std::cout << std::endl << "Max nb of config reached" << std::endl;
  stop_register = true;
 }

 std::vector<double> config;
 for (int i = 0; i < order; ++i) {
  config.emplace_back(get_config(i).t);
 }

 if (config_list.size() == 0) {
  config_list.emplace_back(config);
  config_weight.emplace_back(1);
  return;
 }

 if (config == config_list[config_list.size() - 1])
  config_weight[config_weight.size() - 1]++;
 else {
  config_list.emplace_back(config);
  config_weight.emplace_back(1);
 }
};

// -----------------------
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
