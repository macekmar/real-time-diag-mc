include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

list(APPEND test_names 
	cofactor_row_1
	cofactor_row_2
	configuration_1
	configuration_2
	configuration_3
	g0_keldysh_alpha
	kernel_binning_1
	kernel_binning_2
	kernel_binning_3)

foreach(test_name ${test_names})
	set(test_file "${test_name}.x")
	add_executable(${test_file} "${test_name}.cpp")
	target_link_libraries(${test_file} ctint_keldysh_c ${TRIQS_LIBRARY_ALL})
	triqs_set_rpath_for_target(${test_file})
	add_test(${test_name} ${test_file})
endforeach()

