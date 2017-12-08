include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

add_executable(gray_test.x gray_test.cpp)
target_link_libraries(gray_test.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(gray_test.x)
add_test(gray_test gray_test.x)

add_executable(kernel_binning_1.x kernel_binning_1.cpp)
target_link_libraries(kernel_binning_1.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(kernel_binning_1.x)
add_test(kernel_binning_1 kernel_binning_1.x)

add_executable(kernel_binning_2.x kernel_binning_2.cpp)
target_link_libraries(kernel_binning_2.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(kernel_binning_2.x)
add_test(kernel_binning_2 kernel_binning_2.x)

add_executable(cofactor_row_1.x cofactor_row_1.cpp)
target_link_libraries(cofactor_row_1.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(cofactor_row_1.x)
add_test(cofactor_row_1 cofactor_row_1.x)

add_executable(cofactor_row_2.x cofactor_row_2.cpp)
target_link_libraries(cofactor_row_2.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(cofactor_row_2.x)
add_test(cofactor_row_2 cofactor_row_2.x)

add_executable(g0_keldysh_alpha.x g0_keldysh_alpha.cpp)
target_link_libraries(g0_keldysh_alpha.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(g0_keldysh_alpha.x)
add_test(g0_keldysh_alpha g0_keldysh_alpha.x)

add_executable(configuration_1.x configuration_1.cpp)
target_link_libraries(configuration_1.x ctint_keldysh_c ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(configuration_1.x)
add_test(configuration_1 configuration_1.x)

add_executable(configuration_2.x configuration_2.cpp)
target_link_libraries(configuration_2.x ctint_keldysh_c ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(configuration_2.x)
add_test(configuration_2 configuration_2.x)

