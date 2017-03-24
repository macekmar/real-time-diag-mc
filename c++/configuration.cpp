#include "./configuration.hpp"

Configuration::Configuration(g0_keldysh_t green_function, const keldysh_contour_pt tau,
                             const keldysh_contour_pt taup, const int op_to_measure_spin,
                             array<double, 1> weight_offsets, double weight_blur_time, int max_order)
   : op_to_measure_spin(op_to_measure_spin), order(0), weight_offsets(weight_offsets), weight_blur_time(weight_blur_time), max_order(max_order) {

 weight_sum = array<double, 1>(max_order + 1);
 weight_sum() = 0;
 nb_values = array<int, 1>(max_order + 1);
 nb_values() = 0;

 if (first_dim(weight_offsets) <= max_order) 
  TRIQS_RUNTIME_ERROR << "There is not enough offset values !";

 // Initialize the M-matrices. 100 is the initial alocated space.
 for (auto spin : {up, down}) matrices.emplace_back(green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);

 matrices[op_to_measure_spin].insert_at_end(tau, taup);

 // Initialize weight value
 weight_value = recompute_sum_keldysh_indices(matrices, 0);
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

void Configuration::change_left_input(keldysh_contour_pt tau) {
 matrices[op_to_measure_spin].change_row(matrices[op_to_measure_spin].size() - 1, tau);
};

void Configuration::change_right_input(keldysh_contour_pt taup) {
 matrices[1 - op_to_measure_spin].change_col(matrices[1 - op_to_measure_spin].size() - 1, taup);
};

keldysh_contour_pt Configuration::get_config(int p) {
 return matrices[0].get_x(p);
}; // assuming alpha is a single point

keldysh_contour_pt Configuration::get_left_input() {
 return matrices[op_to_measure_spin].get_x(matrices[op_to_measure_spin].size() - 1);
};

keldysh_contour_pt Configuration::get_right_input() {
 return matrices[op_to_measure_spin].get_y(matrices[op_to_measure_spin].size() - 1);
};

dcomplex Configuration::weight_evaluate() {
 double value = std::abs(recompute_sum_keldysh_indices(matrices, matrices[1 - op_to_measure_spin].size()));
 weight_sum(order) += value;
 nb_values(order)++;

 if (weight_offsets(order) < 0 or value < weight_offsets(order)) {
  keldysh_contour_pt tau_save = get_left_input();
  keldysh_contour_pt tau = tau_save;
  tau.t += weight_blur_time;
  change_left_input(tau);
  value += std::abs(recompute_sum_keldysh_indices(matrices, matrices[1 - op_to_measure_spin].size()));
  change_left_input(tau_save);
 }

 return value;
};

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
 for (int i = 0; i < matrices[1 - op_to_measure_spin].size(); ++i) {
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
