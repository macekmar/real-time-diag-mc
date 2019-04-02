# Generated automatically using the command :
# c++2py.py ../c++/sobol.hpp
from wrap_generator import *

# The module
module = module_(full_name = "sobol", doc = "", app_name = "sobol")

# All the triqs C++/Python modules

# Add here all includes beyond what is automatically included by the triqs modules
module.add_include("sobol.hpp")

# Add here anything to add in the C++ code at the start, e.g. namespace using
module.add_preamble("""
#include <triqs/python_tools/converters/arrays.hpp>
#include <triqs/python_tools/converters/tuple.hpp>
""")
module.add_function ("int i4_bit_hi1 (int n)", doc = """""")

module.add_function ("int i4_bit_lo0 (int n)", doc = """""")

module.add_function ("std::tuple<int,array<double,1>> i4_sobol (int dim_num, int seed)", doc = """""")

module.add_function ("array<double,2> i4_sobol_generate (int m, int n, int skip)", doc = """""")

module.generate_code()