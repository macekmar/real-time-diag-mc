#include "moves_measures.hpp"
#include <triqs/det_manip.hpp>
using triqs::det_manip::det_manip;


// ------------ QMC insertion move --------------------------------------

dcomplex move_insert::attempt() {
  
  auto k = config->perturbation_order(); // order before adding a time
  quick_exit = (k >= params->max_perturbation_order);
  if (quick_exit) return 0;
  
  // insert the new line and col.
  int rx = rng(params->L); // x coordinate
  int ry = rng(params->L); // y coordinate
  auto r = mindex(rx,ry,0);
  auto tau = config->tau_seg.get_random_pt(rng); // new time
  for (auto &m : config->matrices) m.insert_at_end({r,tau, 0}, {r,tau, 0});
  sum_dets = config->recompute_sum_keldysh_indices();
  
  // The Metropolis ratio
  return params->tmax * params->L*params->L * params->U / (k+1) * sum_dets / config->sum_keldysh_indices;
}

dcomplex move_insert::accept() {
  config->sum_keldysh_indices = sum_dets;
  return 1.0;
}

void move_insert::reject() {
  if (quick_exit) return;
  for (auto &m : config->matrices) m.remove(m.size() - 1, m.size() - 1);
}


// ------------ QMC double-insertion move --------------------------------------

dcomplex move_insert2::attempt() {
  auto k = config->perturbation_order(); // order before adding two times
  quick_exit = (k+2 > params->max_perturbation_order);
  if (quick_exit){
    return 0;
  }
  
  // insert the new lines and cols.
  int rx1 = rng(params->L); // x coordinate1
  int ry1 = rng(params->L); // y coordinate1
  int rx2 = rng(params->L); // x coordinate2
  int ry2 = rng(params->L); // y coordinate2
  auto r1 = mindex(rx1,ry1,0);
  auto r2 = mindex(rx2,ry2,0);
  auto tau1 = config->tau_seg.get_random_pt(rng); // new time1
  auto tau2 = config->tau_seg.get_random_pt(rng); // new time2
  for (auto &m : config->matrices) m.insert2_at_end({r1, tau1, 0}, {r2, tau2, 0}, {r1, tau1, 0}, {r2, tau2, 0});
  sum_dets = config->recompute_sum_keldysh_indices();
  
  // The Metropolis ratio
  return pow(params->tmax * params->L*params->L * params->U,2) / ((k+1)*(k+2)) * sum_dets / config->sum_keldysh_indices;
}

dcomplex move_insert2::accept() {
  config->sum_keldysh_indices = sum_dets;
  return 1.0;
}

void move_insert2::reject() {
  if (quick_exit) return;
  for (auto &m : config->matrices) m.remove2_at_end();
}


// ------------ QMC removal move --------------------------------------

dcomplex move_remove::attempt() {
  
  auto k = config->perturbation_order(); // order before removal
  quick_exit = (k <= params->min_perturbation_order);
  if (quick_exit) return 0;
  
  // remove the line/col
  int p = rng(k); // Choose one of the operators for removal
  removed_pt = config->matrices[down].get_x(p);
  config->matrices[down].remove(p, p);
  config->matrices[up].remove(p + 1, p + 1);
  sum_dets = config->recompute_sum_keldysh_indices();
  
  // The Metropolis ratio
  return k / (params->tmax * params->L*params->L * params->U) * sum_dets / config->sum_keldysh_indices;
}

dcomplex move_remove::accept() {
  config->sum_keldysh_indices = sum_dets;
  return 1.0;
}

void move_remove::reject() {
  if (quick_exit) return;
  for (auto &m : config->matrices) m.insert_at_end(removed_pt, removed_pt);
}


// ------------ QMC double-removal move --------------------------------------

dcomplex move_remove2::attempt() {
  
  auto k = config->perturbation_order(); // order before removal
  quick_exit = (k-2 < params->min_perturbation_order);
  if (quick_exit) return 0;
  
  // remove the lines/cols
  int p1 = rng(k); // Choose one of the operators for removal
  int p2 = rng(k-1);
  if (p2 >= p1) p2++;
  removed_pt1 = config->matrices[down].get_x(p1);
  removed_pt2 = config->matrices[down].get_x(p2);
  config->matrices[down].remove2(p1, p2, p1, p2);
  config->matrices[up].remove2(p1+1, p2+1, p1+1, p2+1);
  sum_dets = config->recompute_sum_keldysh_indices();
  
  // The Metropolis ratio
  return k*(k-1) / pow(params->tmax * params->L*params->L * params->U,2) * sum_dets / config->sum_keldysh_indices;
}

dcomplex move_remove2::accept() {
  config->sum_keldysh_indices = sum_dets;
  return 1.0;
}

void move_remove2::reject() {
  if (quick_exit) return;
  for (auto &m : config->matrices) m.insert2_at_end(removed_pt1, removed_pt2, removed_pt1, removed_pt2);
}
