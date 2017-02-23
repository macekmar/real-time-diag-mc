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
| n_cycles                | int                                              |                                                | Number of QMC cycles                                                    |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| length_cycle            | int                                              | 50                                             | Length of a single QMC cycle                                            |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| n_warmup_cycles         | int                                              | 5000                                           | Number of cycles for thermalization                                     |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| random_seed             | int                                              | 34788+928374*triqs::mpi::communicator().rank() | Seed for random number generator                                        |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| random_name             | std::string                                      | ""                                             | Name of random number generator                                         |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| max_time                | int                                              | -1                                             | Maximum runtime in seconds, use -1 to set infinite                      |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| verbosity               | int                                              | ((triqs::mpi::communicator().rank()==0)?3:0)   | Verbosity level                                                         |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+
| method                  | int                                              | 4                                              | Method                                                                  |
+-------------------------+--------------------------------------------------+------------------------------------------------+-------------------------------------------------------------------------+ """)

c.add_method("""void set_g0 (gf_view<retime,matrix_valued> g0_lesser, gf_view<retime,matrix_valued> g0_greater)""",
             doc = """ """)

c.add_method("""int run (int max_time)""",
             doc = """ """)

c.add_property(name = "order_zero",
               getter = cfunction("int order_zero ()"),
               doc = """ """)

c.add_property(name = "solve_duration",
               getter = cfunction("double get_solve_duration ()"),
               doc = """ """)

c.add_property(name = "nb_measures",
               getter = cfunction("int get_nb_measures ()"),
               doc = """ """)

c.add_property(name = "config_list",
               getter = cfunction("std::vector<std::vector<double>> get_config_list ()"),
               doc = """ """)

c.add_property(name = "config_weight",
               getter = cfunction("std::vector<int> get_config_weight ()"),
               doc = """ """)

c.add_property(name = "pn",
               getter = cfunction("array<double,1> get_pn ()"),
               doc = """ """)

c.add_property(name = "sn",
               getter = cfunction("array<dcomplex,3> get_sn ()"),
               doc = """ """)

c.add_property(name = "pn_errors",
               getter = cfunction("array<double,1> get_pn_errors ()"),
               doc = """ """)

c.add_property(name = "sn_errors",
               getter = cfunction("array<double,1> get_sn_errors ()"),
               doc = """ """)

module.add_class(c)

module.generate_code()