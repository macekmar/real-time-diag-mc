# Generated automatically using the command :
# c++2py.py ../c++/g0_semi_circ.hpp
from wrap_generator import *

# The module
module = module_(full_name = "g0_semi_circ", doc = "", app_name = "g0_semi_circ")

# All the triqs C++/Python modules
module.use_module('gf', 'triqs')

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("g0_semi_circ.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <triqs/python_tools/converters/pair.hpp>
""")
module.add_function ("std::pair<gf_view<retime>,gf_view<retime>> make_g0_semi_circular (double beta, double Gamma, double tmax_gf0, int Nt_gf0, double epsilon_d, double muL, double muR)", doc = """""")

module.generate_code()

