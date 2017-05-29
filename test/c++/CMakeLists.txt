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

