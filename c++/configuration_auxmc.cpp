#include "./qmc_data.hpp"
#include "./configuration.hpp"
#include <triqs/det_manip.hpp>


#include <forward_list>
#include <set>
#include <iterator>
#include <utility>
#include <list>

ConfigurationAuxMC::ConfigurationAuxMC(g0_keldysh_alpha_t green_function, const solve_parameters_t &params) 
 : Configuration(green_function, params), 
   config_qmc(green_function, params) 
{
 zero_weight = config_qmc.current_weight;
 // Configuration constructor calls evaluate, but zero_weight is defined
 // only with config_qmc. We have to re-evaluate to get corret
 // `current_weight` and `accepted_weight`.
 evaluate();
 accept_config();
};

void ConfigurationAuxMC::evaluate() {
 // std::cout << "Aux eval" << std::endl;
 dcomplex prod;
 vertex_t vtx;  
 wrapped_forward_list<vertex_t> vertices = vertices_list_;
 wrapped_forward_list<vertex_t> eval_vertices = {};
 wrapped_forward_list<vertex_t>::iterator vi;


 // Calculate
 if (order == 0) {
  current_weight = zero_weight;}
 else {
  // Marjan: TODO: orbitals instead of zeros? What about spin?
 vertices.sort(compare_vertices);

 // std::cout << "Times = ";
 // for (vi = vertices.begin(); vi != vertices.end(); ++vi) {
 //  std::cout << vi->t << " " ;
 // };
 // std::cout << std::endl;

 // std::cout << "Pairs :"; 
 
  vtx = vertices.front();
  vtx = {0, 0, vtx.t, 0, pot_data.potential_of(0, 0)};
  eval_vertices.insert(0, vtx);
  _eval(eval_vertices);
  prod = config_qmc.current_weight;
  eval_vertices.clear();

  // std::cout << "(" << vtx.t << ")";
  // std::cout<<"First " << config_qmc.current_weight << std::endl;

  for (vi = vertices.begin(); vi != vertices.end(); ++vi) {
   if (std::next(vi, 1) == vertices.end()) break;
   vtx = {0, 0, std::next(vi, 1)->t - vi->t, 0, pot_data.potential_of(0, 0)};
   eval_vertices.insert(0, vtx);
   _eval(eval_vertices);
   prod *= config_qmc.current_weight;
   
   eval_vertices.clear();  

   // std::cout << "(" << vtx.t << ")";
   // std::cout<<"Second " << config_qmc.current_weight << std::endl;
  }

  current_weight = std::abs(prod);

 }
}

void ConfigurationAuxMC::_eval(wrapped_forward_list<vertex_t> vertices) {
 potential = 1.0;
 config_qmc.reset_to_vertices(vertices);
 config_qmc.evaluate();
}
