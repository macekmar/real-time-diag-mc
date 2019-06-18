#include "../c++/parameters.hpp"
#include "../c++/configuration.hpp"
#include "../c++/solver_core.hpp"
#include "../c++/g0_flat_band.hpp"
#include <triqs/mc_tools/random_generator.hpp>
#include <mpi.h>

int main() {

MPI::Init(); // needed to create parameters (because of seed)

std::cout << "Here" << std::endl;

int order = 7;
int num_samples = 100;

std::vector<std::vector<double>> intervals(order, {0., 0.0037, 0.0115, 0.0245, 0.044, 0.0718, 0.1117, 0.1666, 0.3253, 1.});
std::vector<std::vector<std::vector<double>>> coeff(order, {{100. , 95.7, 91.4, 88.9}, {88.9, 83.7, 80.6, 77.8}, {77.8, 73. , 69.7, 66.7}, {66.7, 62.2, 58.6, 55.6}, {55.6, 51.2, 47.5, 44.4}, {44.4, 40. , 36.5, 33.3}, {33.3, 29. , 24.3, 22.2}, {22.2, 16.2, 12.7, 11.1}, {11.1, 4.2, 0., 0.}});
       
// prameters
 solve_parameters_t params;

 params.alpha = 0.5;
 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
 
 params.extern_alphas = {0.0};
 params.nb_orbitals = 1;

 
 params.interaction_start = 100.;
  
 std::get<0>(params.potential) = {1.};
 std::get<1>(params.potential) = {0};
 std::get<2>(params.potential) = {0};

 params.U = {1,1,1,1,1,1,1,1,1,1};
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.0;
 params.w_shift = 0.0;
 params.max_perturbation_order = 10;
 params.min_perturbation_order = 0;
 params.method = 1;
 params.singular_thresholds = std::pair<double, double>{3.5, 3.5};

 params.sampling_model_coeff = coeff;
 params.sampling_model_intervals = intervals;
 

 solver_core solver(params);
 
 
 auto g0 = make_g0_flat_band(10000.0, 0.2, 100.0, 10000, 0.2, 0.0, 0.0);
 gf_view<retime> g0_less = g0.first;
 gf_view<retime> g0_grea = g0.second;
 
 auto g0_l = gf<retime>{{-100, 100, 2 * 10000 - 1}, {1, 1}};
 auto g0_g = gf<retime>{{-100, 100, 2 * 10000 - 1}, {1, 1}};
 

for (auto t: g0_l.mesh()) {
   g0_l[t] = g0_grea(t);
 }

 for (auto t: g0_g.mesh()) {
   g0_g[t] = g0_grea(t);
 }

 solver.set_g0(g0_l, g0_g);
 solver.set_model(intervals, coeff);
 solver.init_measure(1);

 triqs::mc_tools::random_generator RNG("mt19937", 23432);


 bool found_sample;

 std::vector<timec_t> l;

 for (int i = 0; i < num_samples; i++) {
  std::cout << "Calculating sample " << i << std::endl;
  
  
  found_sample = false;
  while (!found_sample) {
   
   l.clear();
   for (int j = 0; j < order; j++) l.push_back(RNG());

   auto u = solver.l_to_u(l);
   found_sample = true;
   for (auto u_i = u.begin(); u_i != u.end(); u_i++) {
    if (*u_i < -100.0) found_sample = false;
   }
   std::cout << std::endl;
  }
 
  solver.evaluate_importance_sampling(l);
  
 }



 return 0;
}