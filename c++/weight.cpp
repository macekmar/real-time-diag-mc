#include "./weight.hpp"

// ---------- two_det_weight --------------


two_det_weight::two_det_weight(const solve_parameters_t* params, const input_physics_data* physics_params) {

 // Initialize the M-matrices. 100 is the initial alocated space.
 for (auto spin : {up, down}) matrices.emplace_back(physics_params->green_function, 100);
 for (auto spin : {up, down}) matrices[spin].set_singular_threshold(singular_threshold);

 op_to_measure_spin = physics_params->op_to_measure_spin; // for now
 matrices[op_to_measure_spin].insert_at_end(physics_params->tau_list[0], physics_params->taup);

 // Initialize value
 value = recompute_sum_keldysh_indices(matrices, 0);
}

void two_det_weight::insert(int k, keldysh_contour_pt pt) {
 for (auto& m : matrices) m.insert(k, k, pt, pt);
};

void two_det_weight::insert2(int k1, int k2, keldysh_contour_pt pt1, keldysh_contour_pt pt2) {
 for (auto& m : matrices) m.insert2(k1, k2, k1, k2, pt1, pt2, pt1, pt2);
};

void two_det_weight::remove(int k) {
 for (auto& m : matrices) m.remove(k, k);
};

void two_det_weight::remove2(int k1, int k2) {
 for (auto& m : matrices) m.remove2(k1, k2, k1, k2);
};

void two_det_weight::change_config(int k, keldysh_contour_pt pt) {
 // for (auto& m : matrices) m.change_one_row_and_one_col(k, k, pt, pt);
 for (auto& m : matrices) {
  m.change_row(k, pt);
  m.change_col(k, pt);
 };
};

void two_det_weight::change_left_input(keldysh_contour_pt tau) {
 matrices[op_to_measure_spin].change_row(matrices[op_to_measure_spin].size() - 1, tau);
};

void two_det_weight::change_right_input(keldysh_contour_pt taup) {
 matrices[1 - op_to_measure_spin].change_col(matrices[1 - op_to_measure_spin].size() - 1, taup);
};

keldysh_contour_pt two_det_weight::get_config(int p) { return matrices[0].get_x(p); }; // assuming alpha is a single point

keldysh_contour_pt two_det_weight::get_left_input() {
 return matrices[op_to_measure_spin].get_x(matrices[op_to_measure_spin].size() - 1);
};

keldysh_contour_pt two_det_weight::get_right_input() {
 return matrices[op_to_measure_spin].get_y(matrices[op_to_measure_spin].size() - 1);
};

dcomplex two_det_weight::evaluate() { return recompute_sum_keldysh_indices(matrices, matrices[1 - op_to_measure_spin].size()); };

void two_det_weight::register_config() {
 if (stop_register) return;

 int threshold = 1e7 / triqs::mpi::communicator().size();
 if (config_list.size() > threshold) {
  if (triqs::mpi::communicator().rank() == 0) std::cout << std::endl << "Max nb of config reached" << std::endl;
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
