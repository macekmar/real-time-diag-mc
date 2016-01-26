# where will the python end up in triqs?
set(python_destination applications/impurity_solvers/ctint_new)

# make a local pytriqs copy
triqs_prepare_local_pytriqs_merged_with_my_python(${python_destination})

# Build the cython module
triqs_python_extension(ctint_new ${python_destination})
target_link_libraries(ctint_new ctint_new_c)
include_directories(${TRIQS_INCLUDE_ALL} ${CMAKE_CURRENT_SOURCE_DIR})
triqs_set_rpath_for_target(ctint_new)

# to be able to run from toplevel
execute_process(COMMAND ln -fs ${CMAKE_CURRENT_BINARY_DIR}/ctint_new.so ${CMAKE_BINARY_DIR}/pytriqs/${python_destination})

