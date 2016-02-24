# make a local pytriqs copy
triqs_prepare_local_pytriqs(${CMAKE_PROJECT_NAME})

# Build the cython module
triqs_python_extension_and_install(ctint_keldysh)
target_link_libraries(ctint_keldysh ctint_keldysh_c)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(ctint_keldysh)

# add version file
configure_file(version.py.in version.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.py DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/${CMAKE_PROJECT_NAME})
install (FILES ${PYTHON_SOURCES} DESTINATION ${TRIQS_PYTHON_LIB_DEST_ROOT}/${CMAKE_PROJECT_NAME})
