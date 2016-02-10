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
#include <triqs/python_tools/converters/pair.hpp>
using namespace triqs::gfs;
#include "./converters.hxx"
""")

# The class solver_core
c = class_(
        py_type = "SolverCore",  # name of the python class
        c_type = "solver_core",   # name of the C++ class
        doc = r"",   # doc of the C++ class
)

c.add_constructor("""(gf_view<retime,scalar_valued> g0_lesser, gf_view<retime,scalar_valued> g0_greater)""",
                  doc = """ """)

c.add_method("""std::pair<array<double,1>,array<double,1>> solve (**solve_parameters_t)""",
             doc = """+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| Parameter Name         | Type        | Default                                        | Documentation                                          |
+========================+=============+================================================+========================================================+
| U                      | double      |                                                | U                                                      |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| tmax                   | double      |                                                | tmax                                                   |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| alpha                  | double      |                                                | Alpha term                                             |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| p_dbl                  | double      | 0.5                                            | probability to jump by 2 orders (insert2 and remove2)  |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| max_perturbation_order | int         | 3                                              | Maximum order in U                                     |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| min_perturbation_order | int         | 0                                              | Minimal order in U                                     |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| n_cycles               | int         |                                                | Number of QMC cycles                                   |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| length_cycle           | int         | 50                                             | Length of a single QMC cycle                           |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| n_warmup_cycles        | int         | 5000                                           | Number of cycles for thermalization                    |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| random_seed            | int         | 34788+928374*triqs::mpi::communicator().rank() | Seed for random number generator                       |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| random_name            | std::string | ""                                             | Name of random number generator                        |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| max_time               | int         | -1                                             | Maximum runtime in seconds, use -1 to set infinite     |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+
| verbosity              | int         | ((triqs::mpi::communicator().rank()==0)?3:0)   | Verbosity level                                        |
+------------------------+-------------+------------------------------------------------+--------------------------------------------------------+ """)

module.add_class(c)

module.add_function ("std::pair<gf_view<retime>,gf_view<retime>> make_g0_semi_circular (double beta, double Gamma, double tmax_gf0, int Nt_gf0, double epsilon_d, double muL, double muR)", doc = """""")

module.generate_code()
