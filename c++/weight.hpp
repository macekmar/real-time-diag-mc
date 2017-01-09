#pragma once

struct qmc_weight {

 std::vector<det_manip<g0_keldysh_t>> matrices; // M matrices for up and down
 dcomplex value;                                // Sum of determinants of the last accepted config
 int perturbation_order = 0;                    // the current perturbation order

 qmc_weight(const qmc_weight&) = delete;     // non construction-copyable
 void operator=(const qmc_weight&) = delete; // non copyable

 // ------------
 qmc_weight(const solve_parameters_t* params, g0_keldysh_t* green_function) {

  // Initialize the M-matrices. 100 is the initial alocated space.
  for (auto spin : {up, down}) matrices.emplace_back(*green_function, 100);

  for (auto spin : {up, down}) {
   auto const& ops = params->op_to_measure[spin];
   if (ops.size() == 2) {
    matrices[spin].insert_at_end(make_keldysh_contour_pt(ops[0], params->ref_times.first),
                                 make_keldysh_contour_pt(ops[1], params->ref_times.second));
   }
  }

  value = recompute_sum_keldysh_indices(matrices, 0);
 }
};


struct qmc_weight_single_det {

 det_manip<g0_keldysh_t> matrix; // M matrix
 dcomplex value;                 // Sum of determinants of the last accepted config
 int perturbation_order = 0;     // the current perturbation order

 qmc_weight_single_det(const qmc_weight_single_det&) = delete; // non construction-copyable
 void operator=(const qmc_weight_single_det&) = delete;        // non copyable

 // ------------
 qmc_weight_single_det(const solve_parameters_t* params, g0_keldysh_t* green_function) : matrix(*green_function, 100) {
  // Initialize the M-matrix. 100 is the initial alocated space.

  for (auto spin : {up, down}) {
   auto const& ops = params->op_to_measure[spin];
   if (ops.size() == 2) {
    matrix.insert_at_end(make_keldysh_contour_pt(ops[0], params->ref_times.first),
                         make_keldysh_contour_pt(ops[1], params->ref_times.second));
   }
  }

  value = recompute_sum_keldysh_indices(matrix, 0);
 }
};
