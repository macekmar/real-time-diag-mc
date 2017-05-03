#include "./configuration.hpp"

Configuration::Configuration(g0_keldysh_alpha_t green_function, const keldysh_contour_pt tau,
                             const keldysh_contour_pt taup,
                             int max_order, double singular_threshold)
   : singular_threshold(singular_threshold),
     order(0),
     max_order(max_order) {

 current_kernels = array<dcomplex, 2>(max_order, 2);
 current_kernels() = 0;
 accepted_kernels = array<dcomplex, 2>(max_order, 2);
 accepted_kernels() = 0;


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
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);

 matrices[0].insert_at_end(tau, taup); // first matrix is the big one

 // Initialize weight value
 weight_value = weight_evaluate();
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

void Configuration::change_left_input(keldysh_contour_pt tau) { matrices[0].change_row(order, tau); };

void Configuration::change_right_input(keldysh_contour_pt taup) { matrices[0].change_col(order, taup); };

keldysh_contour_pt Configuration::get_config(int p) const {
 return matrices[0].get_x(p);
}; // assuming alpha is a single point

keldysh_contour_pt Configuration::get_left_input() const { return matrices[0].get_x(order); };

keldysh_contour_pt Configuration::get_right_input() const { return matrices[0].get_y(order); };

// -----------------------
void Configuration::kernels_evaluate_inverse() {
 if (order > 63) TRIQS_RUNTIME_ERROR << "order overflow";

 assert(order > 0); // no kernel for order zero

#ifdef REGENERATE_MATRIX_BEFORE_EACH_GRAY_CODE
 matrices[0].regenerate();
 matrices[1].regenerate();
#endif

 current_kernels() = 0;
 dcomplex det1, det0, inv_value;
 int ap[64] = {0};

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
   int signs[2] = {1, -1};
   keldysh_contour_pt tau = get_left_input();
   keldysh_contour_pt alpha_tmp;
   keldysh_contour_pt alpha = matrices[0].get_y(0);
   matrices[0].remove(order, 0);
   for (int p = 0; p < order; ++p) {
    det0 = matrices[0].determinant();
    current_kernels(p, ap[p]) += signs[(p + order) % 2] * matrices[0].determinant() * det1;
    alpha_tmp = matrices[0].get_y(p);
    matrices[0].change_col(p, alpha);
    alpha = alpha_tmp;
   }
   matrices[0].insert(order, order, tau, alpha);
  } else {
   nb_inverse(order - 1)++;
   for (int p = 0; p < order; ++p) {
    inv_value = matrices[0].inverse_matrix(p, order);
	current_kernels(p, ap[p]) += inv_value * det0 * det1;
   }
  }
  sign = -sign;
 }

};

// -----------------------
// Deprecated
// TODO: write down the formula this implements
void Configuration::kernels_evaluate_cofact() {
 assert(order > 0); // no kernel for order zero

 auto tau = get_left_input();
 auto taup = get_right_input();
 int signs[2] = {1, -1};
 matrices[0].remove(order, order); // remove tau_w and taup

 keldysh_contour_pt alpha_p_right;
 keldysh_contour_pt alpha_tmp = taup;
 matrices[0].roll_matrix(det_manip<g0_keldysh_alpha_t>::RollDirection::Left);

 for (int p = 0; p < order; ++p) {
  alpha_p_right = matrices[0].get_y((p - 1 + order) % order);
  for (int k_index : {1, 0}) {
   alpha_p_right = flip_index(alpha_p_right);
   // matrices[0].change_one_row_and_one_col(p, (p - 1 + order) % order, flip_index(matrices[0].get_x(p)),
   //                                     alpha_tmp);
   // Change the p keldysh index on the left (row) and the p point on the right (col).
   // The p point on the right is effectively changed only when k_index=1.
   matrices[0].change_row(p, flip_index(matrices[0].get_x(p)));
   matrices[0].change_col((p - 1 + order) % order, alpha_tmp);

   // matrices[1].change_one_row_and_one_col(p, p, flip_index(matrices[1].get_x(p)),
   // flip_index(matrices[1].get_y(p)));
   matrices[1].change_row(p, flip_index(matrices[1].get_x(p)));
   matrices[1].change_col(p, flip_index(matrices[1].get_y(p)));

   current_kernels(p, k_index) = keldysh_sum_cofact(p) * signs[(order + p + k_index) % 2];
  }
  alpha_tmp = alpha_p_right;
 }
 matrices[0].change_col(order - 1, alpha_tmp);
 matrices[0].insert(order, order, tau, taup);

};

// -----------------------
double Configuration::weight_kernels() {
 if (order == 0) return 1.; // is this a good value ?
 kernels_evaluate_inverse();

 return sum(abs( current_kernels(range(0, order), range()) ));
};

// -----------------------
dcomplex Configuration::weight_evaluate() {
 double value = weight_kernels();
 weight_sum(order) += value;
 nb_values(order)++;
 return value;
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
 config.emplace_back(get_left_input().t);
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
