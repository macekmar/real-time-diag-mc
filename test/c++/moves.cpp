#include "../c++/parameters.hpp"
#include "../c++/configuration.hpp"
#include "../c++/moves.hpp"
#include "../c++/random_vertex_gen.hpp"
#include <mpi.h>

using namespace triqs::gfs;

bool operator!=(const vertex_t& a, const vertex_t& b) {
 return (a.x_up != b.x_up) or (a.x_do != b.x_do) or (a.t != b.t) or
        (a.k_index != b.k_index) or (a.potential != b.potential);
};

/**
 * Tests wether rejection of moves brings the configuration back to its
 * original configuration. This test relies on Configuration::get_vertex,
 * which should be correct.
 *
 * /!\ This test relies on a random number generator. This is a problem as
 * removed vertices are taken randomly, hence the loop below. However this is
 * not a problem for generation of times (generated double numbers are all
 * different), it can be for orbitals.
 */
int main() {

 MPI::Init(); // needed to create parameters (because of seed)

 auto g_less = gf<retime, matrix_valued>{{-100., 100., 1001}, {3, 3}};
 auto g_less_f = make_gf_from_fourier(g_less);

 // let's build an antihermitian lesser GF.
 for (auto w : g_less_f.mesh()) {
  g_less_f[w](0, 0) = -0.5_j / (w + 1.2);
  g_less_f[w](0, 1) = 1.5_j / (w - 1. + 2.5_j);
  g_less_f[w](0, 2) = 0.5_j / (w - 0.2 + 2.5_j);
  g_less_f[w](1, 0) = -conj(g_less_f[w](0, 1));
  g_less_f[w](1, 1) = -0.7_j / (w - 1.);
  g_less_f[w](1, 2) = 0.9 / (w + 0.8 + 0.5_j);
  g_less_f[w](2, 0) = -conj(g_less_f[w](0, 2));
  g_less_f[w](2, 1) = -conj(g_less_f[w](1, 2));
  g_less_f[w](2, 2) = 1.7_j / (w - 10.);
 }

 g_less = make_gf_from_inverse_fourier(g_less_f);
 auto g_grea = conj(g_less);

 auto g0 = g0_keldysh_t{g_less, g_grea};
 std::vector<dcomplex> alphas{0.2};
 g0_keldysh_alpha_t g0_alpha = g0_keldysh_alpha_t{g0, 0.5, alphas};

 solve_parameters_t params;

 params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 1.5, 0));
 params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 1.5, 0));
 params.extern_alphas = alphas;
 params.nonfixed_op = false;
 params.interaction_start = 50.;
 params.alpha = 0.5;
 params.nb_orbitals = 3;
 std::get<0>(params.potential) = {1., 0.8, 0.5, 0.5};
 std::get<1>(params.potential) = {0, 1, 1, 2};
 std::get<2>(params.potential) = {0, 1, 2, 1};

 params.U = {0.05, 0.05, 0.05};
 params.w_ins_rem = 1.0;
 params.w_dbl = 0.5;
 params.w_shift = 0.5;
 params.max_perturbation_order = 3;
 params.min_perturbation_order = 0;
 params.verbosity = 1;
 params.method = 1;
 params.singular_thresholds = std::pair<double, double>{3.5, 3.5};

 ConfigurationQMC config(g0_alpha, params);

 auto rng = triqs::mc_tools::random_generator("", 87235);
 auto rvg = uniform_rvg(rng, params);
 moves::insert insert(config, params, rng, rvg);
 moves::insert2 insert2(config, params, rng, rvg);
 moves::remove remove(config, params, rng, rvg);
 moves::remove2 remove2(config, params, rng, rvg);
 moves::shift shift(config, params, rng, rvg);

 // test attempt+reject do not change the configuration

 // first insert some vertices
 insert.attempt();
 insert.accept();

 insert.attempt();
 insert.accept();

 insert.attempt();
 insert.accept();

 if (config.order != 3) return 10;

 // remember configuration
 std::vector<vertex_t> vertex_list;
 for (size_t k = 0; k < config.order; ++k)
  vertex_list.push_back(config.get_vertex(k));

  std::cout << "Initial configuration" << std::endl;
 config.print();

 // loop to consider many possibilities
 for (int i = 0; i < 10; ++i) {

  // insert
  std::cout << "insert" << std::endl;
  insert.attempt();
  insert.reject();
  config.print();

  if (config.order != 3) return 11;
  for (size_t k = 0; k < config.order; ++k)
   if (vertex_list[k] != config.get_vertex(k)) return 21;

  // insert2
  std::cout << "insert2" << std::endl;
  insert2.attempt();
  insert2.reject();
  config.print();

  if (config.order != 3) return 12;
  for (size_t k = 0; k < config.order; ++k)
   if (vertex_list[k] != config.get_vertex(k)) return 22;

  // remove
  std::cout << "remove" << std::endl;
  remove.attempt();
  remove.reject();
  config.print();

  if (config.order != 3) return 13;
  for (size_t k = 0; k < config.order; ++k)
   if (vertex_list[k] != config.get_vertex(k)) return 23;

  // remove2
  std::cout << "remove2" << std::endl;
  remove2.attempt();
  remove2.reject();
  config.print();

  if (config.order != 3) return 14;
  for (size_t k = 0; k < config.order; ++k)
   if (vertex_list[k] != config.get_vertex(k)) return 24;

  // shift
  std::cout << "shift" << std::endl;
  shift.attempt();
  shift.reject();
  config.print();

  if (config.order != 3) return 15;
  for (size_t k = 0; k < config.order; ++k)
   if (vertex_list[k] != config.get_vertex(k)) return 25;
 }

 MPI::Finalize();
 return 0;
}
