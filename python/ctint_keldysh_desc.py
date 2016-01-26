# Generated automatically using the command :
# c++2py.py ../c++/ctint.hpp -p -mpytriqs.applications.impurity_solvers.ctint_keldysh -o ctint_keldysh --moduledoc "The ctint solver"
from wrap_generator import *

# The module
module = module_(full_name = "pytriqs.applications.impurity_solvers.ctint_keldysh", doc = "The ctint solver")

# All the triqs C++/Python modules
module.use_module('gf')

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("../c++/ctint.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
using namespace triqs::gfs;
#include "./converters.hxx"
""")

# The class ctint_solver
c = class_(
        py_type = "CtintSolver",  # name of the python class
        c_type = "ctint_solver",   # name of the C++ class
)


c.add_constructor("""(double beta, double mu, int n_freq, double t_min, double t_max, int L, int Lk)""",
                  doc = """ """)

c.add_method("""void solve (**solve_parameters_t)""",
             doc = """  Parameter Name         Type        Default                                        Documentation

  U                      double      --                                             U
  tmax                   double      --                                             tmax
  alpha                  double      --                                             Alpha term
  p_dbl                  double      0.5                                            probability to jump by 2 orders (insert2 and remove2)
  is_current             bool        --                                             Compute <n> or the current.
  max_perturbation_order int         3                                              Maximum order in U
  min_perturbation_order int         0                                              Minimal order in U
  n_cycles               int         --                                             Number of QMC cycles
  length_cycle           int         50                                             Length of a single QMC cycle
  n_warmup_cycles        int         5000                                           Number of cycles for thermalization
  random_seed            int         34788+928374*triqs::mpi::communicator().rank() Seed for random number generator
  random_name            std::string ""                                             Name of random number generator
  max_time               int         -1                                             Maximum runtime in seconds, use -1 to set infinite
  verbosity              int         ((triqs::mpi::communicator().rank()==0)?3:0)   Verbosity level                                       """)

c.add_property(name = "CnSn",
               getter = cfunction("array_view<double,2> CnSn ()"),
               doc = """ """)

c.add_property(name = "c0",
               getter = cfunction("double c0 ()"),
               doc = """ """)

module.add_class(c)

module.generate_code()
