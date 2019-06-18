# Generated automatically using the command :
# c++2py.py ../c++/solver_core.hpp -p -mctint_keldysh -o ctint_keldysh --moduledoc "The ctint solver"
from wrap_generator import *

# The module
module = module_(full_name = "ctint_keldysh", doc = "The ctint solver", app_name = "ctint_keldysh")

# All the triqs C++/Python modules
module.use_module('gf', 'triqs')

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("solver_core.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <triqs/python_tools/converters/arrays.hpp>
#include <triqs/python_tools/converters/pair.hpp>
#include <triqs/python_tools/converters/vector.hpp>
#include <triqs/python_tools/converters/tuple.hpp>
using namespace triqs::gfs;
#include "./ctint_keldysh_converters.hxx"
""")

# The class solver_core
c = class_(
        py_type = "SolverCore",  # name of the python class
        c_type = "solver_core",   # name of the C++ class
        doc = r"",   # doc of the C++ class
)

c.add_constructor("""(**solve_parameters_t)""",
                  doc = """+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| Parameter Name           | Type                                                                             | Default                                        | Documentation                                                                                  |
+==========================+==================================================================================+================================================+================================================================================================+
| creation_ops             | std::vector<std::tuple<orbital_t, int, timec_t, int> >                           |                                                | External Keldysh contour points for the creation operators                                     |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| annihilation_ops         | std::vector<std::tuple<orbital_t, int, timec_t, int> >                           |                                                | External Keldysh contour points for the annihilation operators                                 |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| extern_alphas            | std::vector<dcomplex>                                                            |                                                | External alphas                                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| nonfixed_op              | bool                                                                             | false                                          | Operator to develop upon, in the kernel method.                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| interaction_start        | double                                                                           |                                                | time before 0 at which interaction started                                                     |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| alpha                    | double                                                                           |                                                | Alpha in the density-density interaction term                                                  |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| nb_orbitals              | int                                                                              |                                                | Number of orbitals. Orbitals are indexed between 0 and `nb_orbitals`-1.                        |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| potential                | std::tuple<std::vector<double>, std::vector<orbital_t>, std::vector<orbital_t> > |                                                |                                                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| U                        | std::vector<double>                                                              |                                                | U list, one for each order (including forbidden ones)                                          |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| w_ins_rem                | double                                                                           | 1.0                                            | weight of insert and remove                                                                    |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| w_dbl                    | double                                                                           | 0.5                                            | weight of insert2 and remove2                                                                  |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| w_shift                  | double                                                                           | 0.0                                            | weight of the shift move                                                                       |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| max_perturbation_order   | int                                                                              | 3                                              | Maximum order in U                                                                             |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| min_perturbation_order   | int                                                                              | 0                                              | Minimal order in U                                                                             |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| forbid_parity_order      | int                                                                              | -1                                             | Parity of the orders automatically rejected. -1 (default) to reject no order.                  |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| length_cycle             | int                                                                              | 50                                             | Length of a single QMC cycle                                                                   |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| random_seed              | int                                                                              | 34788+928374*triqs::mpi::communicator().rank() | Seed for random number generator                                                               |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| random_name              | std::string                                                                      | ""                                             | Name of random number generator                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| verbosity                | int                                                                              | ((triqs::mpi::communicator().rank()==0)?3:0)   | Verbosity level                                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| method                   | int                                                                              | 1                                              | Method                                                                                         |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| nb_bins                  | int                                                                              | 10000                                          | nb of bins for the kernels                                                                     |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| singular_thresholds      | std::pair<double, double>                                                        |                                                | log10 conditioning thresholds for each det_manip objects                                       |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| cycles_trapped_thresh    | int                                                                              | 100                                            | Number of cycles after which a trapped configuration is reevaluated                            |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| store_configurations     | int                                                                              | 0                                              | Store the list of all configurations accepted (if 1) or attempted (if 2) in the Markov chain.  |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| preferential_sampling    | bool                                                                             | false                                          | Do preferential sampling or not                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| ps_gamma                 | double                                                                           | 1.                                             | Preferential sampling parameter -- time sampling lorentzian width                              |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| sampling_model_intervals | std::vector<std::vector<double> >                                                |                                                |                                                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+
| sampling_model_coeff     | std::vector<std::vector<std::vector<double> > >                                  |                                                |                                                                                                |
+--------------------------+----------------------------------------------------------------------------------+------------------------------------------------+------------------------------------------------------------------------------------------------+ """)

c.add_method("""void set_g0 (gf_view<retime,matrix_valued> g0_lesser, gf_view<retime,matrix_valued> g0_greater)""",
             doc = """ """)

c.add_method("""int run (int nb_cycles, bool do_measure, int max_time)""",
             doc = """ """)

c.add_method("""int run (int nb_cycles, bool do_measure)""",
             doc = """ """)

c.add_method("""bool collect_results (int size_partition)""",
             doc = """ """)

c.add_method("""dcomplex evaluate_qmc_weight (std::vector<std::tuple<orbital_t,orbital_t,timec_t>> vertices)""",
             doc = """ """)

c.add_method("""std::vector<dcomplex> evaluate_importance_sampling (std::vector<timec_t> times_l, bool do_measure = false)""",
             doc = """ """)

c.add_method("""dcomplex evaluate_model (std::vector<timec_t> times_l_vec)""",
             doc = """ """)

c.add_method("""timec_t inverse_cdf (int i, timec_t l)""",
             doc = """ """)

c.add_method("""void collect_sampling_weights (int dummy)""",
             doc = """ """)

c.add_method("""void set_model (std::vector<std::vector<double>> intervals, std::vector<std::vector<std::vector<double>>> coeff)""",
             doc = """ """)

c.add_method("""std::vector<double> l_to_u (std::vector<timec_t> times_l)""",
             doc = """ """)

c.add_method("""std::vector<double> u_to_l (std::vector<timec_t> times_u)""",
             doc = """ """)

c.add_method("""std::vector<double> l_to_v (std::vector<timec_t> times_l)""",
             doc = """ """)

c.add_method("""std::vector<double> v_to_l (std::vector<timec_t> times_v)""",
             doc = """ """)

c.add_method("""std::vector<double> v_to_u (std::vector<timec_t> times_v)""",
             doc = """ """)

c.add_method("""std::vector<double> u_to_v (std::vector<timec_t> times_u)""",
             doc = """ """)

c.add_method("""void init_measure (int dummy)""",
             doc = """ """)

c.add_property(name = "qmc_duration",
               getter = cfunction("double get_qmc_duration ()"),
               doc = """ """)

c.add_property(name = "nb_measures",
               getter = cfunction("long get_nb_measures ()"),
               doc = """ """)

c.add_property(name = "config_list",
               getter = cfunction("std::vector<std::vector<double>> get_config_list ()"),
               doc = """ """)

c.add_property(name = "config_mult",
               getter = cfunction("std::vector<int> get_config_mult ()"),
               doc = """ """)

c.add_property(name = "config_weight",
               getter = cfunction("std::vector<dcomplex> get_config_weight ()"),
               doc = """ """)

c.add_property(name = "pn",
               getter = cfunction("array<long,1> get_pn ()"),
               doc = """ """)

c.add_property(name = "sn",
               getter = cfunction("array<dcomplex,1> get_sn ()"),
               doc = """ """)

c.add_property(name = "kernels",
               getter = cfunction("array<dcomplex,4> get_kernels ()"),
               doc = """ """)

c.add_property(name = "kernel_diracs",
               getter = cfunction("array<dcomplex,4> get_kernel_diracs ()"),
               doc = """ """)

c.add_property(name = "nb_kernels",
               getter = cfunction("array<long,4> get_nb_kernels ()"),
               doc = """ """)

c.add_property(name = "bin_times",
               getter = cfunction("array<double,1> get_bin_times ()"),
               doc = """ """)

c.add_property(name = "dirac_times",
               getter = cfunction("array<double,1> get_dirac_times ()"),
               doc = """ """)

c.add_property(name = "U",
               getter = cfunction("std::vector<double> get_U ()"),
               doc = """ """)

c.add_property(name = "max_order",
               getter = cfunction("int get_max_order ()"),
               doc = """ """)

module.add_class(c)

module.generate_code()