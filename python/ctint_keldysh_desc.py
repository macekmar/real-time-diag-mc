# Generated automatically using the command :
# c++2py.py ../c++/solver_core.hpp -p -mctint_keldysh -o ctint_keldysh --moduledoc "The ctint solver" --libclang_location /usr/lib/llvm-3.8/lib/libclang-3.8.so
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
                  doc = """+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| Parameter Name          | Type                                             | Default                                        | Documentation                                                           |
+=========================+==================================================+================================================+=========================================================================+
| right_input_points      | std::vector<std::tuple<x_index_t, double, int> > |                                                | input contour points, except the first (left) one, must be of odd size  |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| interaction_start       | double                                           |                                                | time before 0 at which interaction started                              |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| measure_state           | int                                              | 0                                              | measure states (for the first input point), for now just one            |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| measure_times           | std::vector<double>                              |                                                | measure times (for the first input point)                               |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| measure_keldysh_indices | std::vector<int>                                 |                                                | measure keldysh indices (for the first input point)                     |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| alpha                   | double                                           |                                                | Alpha term                                                              |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| U                       | double                                           |                                                | U                                                                       |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| w_ins_rem               | double                                           | 1.0                                            | weight of insert and remove                                             |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| w_dbl                   | double                                           | 0.5                                            | weight of insert2 and remove2                                           |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| w_shift                 | double                                           | 0.0                                            | weight of the shift move                                                |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| w_weight_swap           | double                                           | 0.01                                           | weight of the weight_swap move                                          |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| w_weight_shift          | double                                           | 0.01                                           | weight of the weight_shift move                                         |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| max_perturbation_order  | int                                              | 3                                              | Maximum order in U                                                      |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| min_perturbation_order  | int                                              | 0                                              | Minimal order in U                                                      |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| length_cycle            | int                                              | 50                                             | Length of a single QMC cycle                                            |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| random_seed             | int                                              | 34788+928374*triqs::mpi::communicator().rank() | Seed for random number generator                                        |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| random_name             | std::string                                      | ""                                             | Name of random number generator                                         |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| max_time                | int                                              | -1                                             | Maximum runtime in seconds, use -1 to set infinite                      |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| verbosity               | int                                              | 0                                              | Verbosity level                                                         |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| method                  | int                                              | 5                                              | Method                                                                  |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| nb_bins                 | int                                              | 10000                                          | nb of bins for the kernels                                              |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| singular_thresholds     | std::pair<double, double>                        |                                                | log10 conditioning thresholds for each det_manip objects                |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+ """)

c.add_method("""void set_g0 (gf_view<retime,matrix_valued> g0_lesser, gf_view<retime,matrix_valued> g0_greater)""",
             doc = """ """)

c.add_method("""int run (int nb_cycles, bool do_measure, int max_time)""",
             doc = """ """)

c.add_method("""int run (int nb_cycles, bool do_measure)""",
             doc = """ """)

c.add_property(name = "order_zero",
               getter = cfunction("std::tuple<double,array<dcomplex,2>> order_zero ()"),
               doc = """ """)

c.add_property(name = "compute_sn_from_kernels",
               getter = cfunction("void compute_sn_from_kernels ()"),
               doc = """ """)

c.add_property(name = "solve_duration",
               getter = cfunction("double get_solve_duration ()"),
               doc = """ """)

c.add_property(name = "solve_duration_all",
               getter = cfunction("double get_solve_duration_all ()"),
               doc = """ """)

c.add_property(name = "nb_measures",
               getter = cfunction("long get_nb_measures ()"),
               doc = """ """)

c.add_property(name = "nb_measures_all",
               getter = cfunction("long get_nb_measures_all ()"),
               doc = """ """)

c.add_property(name = "config_list",
               getter = cfunction("std::vector<std::vector<double>> get_config_list ()"),
               doc = """ """)

c.add_property(name = "config_weight",
               getter = cfunction("std::vector<int> get_config_weight ()"),
               doc = """ """)

c.add_property(name = "pn",
               getter = cfunction("array<long,1> get_pn ()"),
               doc = """ """)

c.add_property(name = "pn_all",
               getter = cfunction("array<long,1> get_pn_all ()"),
               doc = """ """)

c.add_property(name = "sn",
               getter = cfunction("array<dcomplex,3> get_sn ()"),
               doc = """ """)

c.add_property(name = "sn_all",
               getter = cfunction("array<dcomplex,3> get_sn_all ()"),
               doc = """ """)

c.add_property(name = "kernels",
               getter = cfunction("array<dcomplex,3> get_kernels ()"),
               doc = """ """)

c.add_property(name = "kernels_all",
               getter = cfunction("array<dcomplex,3> get_kernels_all ()"),
               doc = """ """)

c.add_property(name = "nb_kernels",
               getter = cfunction("array<long,3> get_nb_kernels ()"),
               doc = """ """)

module.add_class(c)

module.add_function ("void compilation_time_stamp (int node_size)", doc = """""")

module.generate_code()