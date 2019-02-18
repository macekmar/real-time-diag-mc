#include "./qmc_data.hpp"
#include "./configuration.hpp"
#include <triqs/det_manip.hpp>

#include <forward_list>
#include <iterator>

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


/**
 * Count the orders for the accepted moves
 */
void ConfigurationAuxMC::accept_config(){
  Configuration::accept_config();
  if (params.print_aux_stats) {++pn[order];}
}

/**
 * Evaluates AuxMC weight by providing list of vertices for _eval
 * 
 * In the first approximation, the AuxMC weight is
 *    W(u_1) * W(u_2 - u_1) * W(u_3 - u_2) * ..., where |u_1| < |u_2| < ...
 * W is the first order QMC weight (`_eval` for one vertex)
 */
void ConfigurationAuxMC::evaluate() {
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
  vertices.sort(compare_vertices_times);

  // First point ...
  vtx = vertices.front();
  vtx = {0, 0, vtx.t, 0, pot_data.potential_of(0, 0)};
  eval_vertices.insert(0, vtx);
  _eval(eval_vertices);
  prod = config_qmc.current_weight;
  eval_vertices.clear();
  // .. and the rest
  for (vi = vertices.begin(); vi != vertices.end(); ++vi) {
   if (std::next(vi, 1) == vertices.end()) break;
   vtx = {0, 0, std::next(vi, 1)->t - vi->t, 0, pot_data.potential_of(0, 0)};
   eval_vertices.insert(0, vtx);
   _eval(eval_vertices);
   prod *= config_qmc.current_weight;
   eval_vertices.clear();  
  }
  current_weight = std::abs(prod);
 }
}

/**
 * Calculates the QMC weight for the list of vertices
 */
void ConfigurationAuxMC::_eval(wrapped_forward_list<vertex_t> vertices) {
 potential = 1.0;
 config_qmc.reset_to_vertices(vertices);
 config_qmc.evaluate();
}
