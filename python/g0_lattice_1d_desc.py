# Generated automatically using the command :
# c++2py.py ../c++/g0_lattice_1d.hpp
from wrap_generator import *

# The module
module = module_(full_name = "g0_lattice_1d", doc = "", app_name = "g0_lattice_1d")

# All the triqs C++/Python modules
module.use_module('gf', 'triqs')

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("g0_lattice_1d.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <triqs/python_tools/converters/pair.hpp>
using namespace triqs::gfs;
""")
module.add_function ("std::pair<gf_view<retime>,gf_view<retime>> make_g0_lattice_1d (double beta, double mu, double epsilon, double hop, double tmax_gf0, int Nt_gf0, int nb_sites, int Nb_k_pts)", doc = """Compute the Green\'s functions of a 1D periodic lattice with nearest-neighboors coupling.\n `beta` is the inverse temperature. If negative, temperature is zero.\n\n TODO: complete description""")

module.generate_code()