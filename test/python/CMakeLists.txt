find_package(TriqsTest)

# Copy h5 files to binary dir
FILE(GLOB all_h5_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h5)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

triqs_add_python_test(density)
triqs_add_python_test(density_order_0)
triqs_add_python_test(correlator)
triqs_add_python_test(correlator_order_0)
#triqs_add_python_test(double_occ)
#triqs_add_python_test(current)
