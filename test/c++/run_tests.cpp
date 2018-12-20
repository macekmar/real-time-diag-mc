#include <exception>
#include <mpi.h>
#include <triqs/arrays.hpp>
#include "../c++/solver_core.hpp"
#include "../c++/g0_flat_band.hpp"
#include "./utility.hpp"

using namespace triqs::gfs;

struct situation {
 const size_t ID;
 const int method;
 const int fpo;
 const double w_ins_rem;
 const double w_dbl;
 const int min_order;
 const bool pref_spl;

 situation(int method, int fpo, double w_ins_rem, double w_dbl, int min_order=0, bool pref_spl=false)
  : ID(nb), method(method), fpo(fpo), w_ins_rem(w_ins_rem), w_dbl(w_dbl), min_order(min_order),
    pref_spl(pref_spl)
 {++nb;};

 private:
  static size_t nb;

 public:
  static size_t get_nb() {return nb;};
};

size_t situation::nb = 0;

/* Check if array(k) is non zero for k >= `start` with a different parity than `parity`
 * (ignored if `parity` < 0), and if other elements are zeros.
 */
template<typename T>
bool check_non_zero(const T &array, size_t start, int parity=-1) {
 bool output = true;
 bool good_parity;
 for (size_t k = 0; k < first_dim(array); ++k) {

  if (parity < 0)
   good_parity = true;
  else
   good_parity = (k % 2) != (parity % 2);

  if (((start <= k) and good_parity)) // orders that should be visited
   output = output and (array(k) != 0.0);
  else // orders that should NOT be visited
   output = output and (array(k) == 0.0);

 }
 return output;
}


/// Integrated test. Test no error occurs for different use cases.
// Also test number of measures reported and if results are not trash.
int main() {

 MPI::Init();

 std::vector<situation> situations;
 situations.emplace_back(0, -1, 1.0, 0.5);
 situations.emplace_back(0, -1, 1.0, 0.5, 1);
 situations.emplace_back(0, 1, 0.0, 1.0);
 situations.emplace_back(0, 0, 0.0, 1.0);
 situations.emplace_back(1, -1, 1.0, 0.5);
 situations.emplace_back(1, -1, 1.0, 0.5, 1);
 situations.emplace_back(1, -1, 1.0, 0.5, 0, true);
 situations.emplace_back(1, 1, 0.0, 1.0);
 situations.emplace_back(2, -1, 1.0, 0.5);
 situations.emplace_back(2, 1, 0.0, 1.0);

 std::map<size_t, bool> success_map;
 bool all_success = true;
 bool success;
 for (auto it = situations.begin(); it != situations.end(); ++it) {
  std::cout << "Trying situation number " << it->ID << std::endl << std::endl;
  success = true;

  try {
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

   solve_parameters_t params;

   params.creation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
   params.annihilation_ops.push_back(std::tuple<orbital_t, spin_t, timec_t, int>(0, up, 0.0, 0));
   params.extern_alphas.push_back(0.);
   params.nonfixed_op = false;
   params.interaction_start = 50.;
   params.alpha = 0.5;
   params.nb_orbitals = 3;
   std::get<0>(params.potential) = {1., 0.8, 0.5, 0.5};
   std::get<1>(params.potential) = {0, 1, 1, 2};
   std::get<2>(params.potential) = {0, 1, 2, 1};

   params.U = std::vector<double>(4 - it->min_order, 0.05);
   params.w_ins_rem = it->w_ins_rem;
   params.w_dbl = it->w_dbl;
   params.w_shift = 0.5;
   params.forbid_parity_order = it->fpo;
   params.max_perturbation_order = 4;
   params.min_perturbation_order = it->min_order;
   params.verbosity = 1;
   params.method = it->method;
   params.singular_thresholds = std::pair<double, double>{4.5, 3.3};
   params.nb_bins = 20;
   params.preferential_sampling = it->pref_spl;

   solver_core S(params);
   S.set_g0(g_less, g_grea);

   S.run(100, false);
   S.run(100, true);
   S.collect_results(1);

   /// check nb of measures
   std::cout << "pn: " << S.get_pn() << std::endl;
   if (S.get_nb_measures() != 100) throw 1;
   S.run(20, true);
   S.collect_results(1);
   std::cout << "pn: " << S.get_pn() << std::endl;
   if (S.get_nb_measures() != 120) throw 2;
   std::cout << "pn: " << S.get_pn() << std::endl;

   /// check pn
   auto pn = S.get_pn();
   if (not all_positive(pn)) throw 10;
   if (not all_finite(pn)) throw 11;

   if (not check_non_zero(pn, it->min_order, it->fpo)) throw 12;

   /// check sn
   if (it->method == 0) {
    auto sn = S.get_sn();
    std::cout << "sn: " << sn << std::endl;
    if (not all_finite(sn)) throw 20;

    if (not check_non_zero(sn, it->min_order, it->fpo)) throw 21;
   }

   /// check kernels
   if (it->method > 0) {
    auto kernels = S.get_kernels();
    std::cout << "kernels: " << kernels << std::endl;
    if (not all_finite(kernels)) throw 30;


    auto kernel_diracs = S.get_kernel_diracs();
    std::cout << "kernel_diracs: " << kernel_diracs << std::endl;
    if (not all_finite(kernel_diracs)) throw 40;

   }

  } catch (std::exception& e) {
   success = false;
   std::cout << e.what() << std::endl;
  } catch (int e) {
   success = false;
   std::cout << "Error code " << e << std::endl;
  } catch (...) {
   success = false;
   std::cout << "Unknown error" << std::endl;
  }
  std::cout << std::endl;
  success_map[it->ID] = success;
  if (not success) all_success = false;

 }

 std::cout << "Summary:" << std::endl;
 for (auto it = success_map.begin(); it != success_map.end(); ++it)
  std::cout << it->first << " : " << (it->second ? "success" : "fail") << std::endl;

 if (success_map.size() != situation::get_nb()) {
  std::cout << "All situations have not been tested!" << std::endl;
  all_success = false;
 }


 MPI::Finalize();

 return all_success ? 0 : 1;
};
