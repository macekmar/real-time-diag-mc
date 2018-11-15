################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Ferrero, O. Parcollet
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

r"""
DOC

"""
from ctint_keldysh import SolverCore
from g0_semi_circ import make_g0_semi_circular, make_g0c_semi_circular_freq, make_g0_semi_circular_freq
from g0_flat_band import make_g0_flat_band, make_g0c_flat_band_freq, make_g0_flat_band_freq
from g0_lattice_1d import make_g0_lattice_1d
from solver import solve, save_configuration_list
from results import merge
from post_treatment import compute_correlator, compute_correlator_oldway, make_g0_contour

__all__ = ['SolverCore',
           'make_g0_semi_circular',
           'make_g0c_semi_circular_freq',
           'make_g0_semi_circular_freq',
           'make_g0_flat_band',
           'make_g0c_flat_band_freq',
           'make_g0_flat_band_freq',
           'make_g0_lattice_1d',
           'solve',
           'save_configuration_list',
           'merge',
           'compute_correlator',
           'compute_correlator_oldway',
           'make_g0_contour']

### Do not ignore DeprecationWarning (revert python weird default)
import warnings
warnings.simplefilter('default', DeprecationWarning)

### Complex numbers are everywhere in this solver, one should not mess with them
import numpy as np
warnings.simplefilter('error', np.ComplexWarning)

