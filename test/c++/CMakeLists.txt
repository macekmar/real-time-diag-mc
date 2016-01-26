include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

add_executable(gray_test.x gray_test.cpp)
target_link_libraries(gray_test.x ctint_new_c ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(gray_test.x)
add_test(gray_test gray_test.x)

add_executable(profile.x profile.cpp)
target_link_libraries(profile.x ctint_new_c ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(profile.x)
