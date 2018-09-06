# make a local pytriqs copy
triqs_prepare_local_pytriqs(${CMAKE_PROJECT_NAME})

# Build the module ctint_keldysh
triqs_python_extension_and_install(ctint_keldysh)
target_link_libraries(ctint_keldysh ctint_keldysh_c)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(ctint_keldysh)

# Build the module g0_semi_circ
triqs_python_extension_and_install(g0_semi_circ)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(g0_semi_circ)

# Build the module g0_flat_band
triqs_python_extension_and_install(g0_flat_band)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(g0_flat_band)

# Build the module g0_lattice_1d
triqs_python_extension_and_install(g0_lattice_1d)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(g0_lattice_1d)

# add version file
configure_file(version.py.in version.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/${CMAKE_PROJECT_NAME})
install (FILES ${PYTHON_SOURCES} DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/${CMAKE_PROJECT_NAME})
