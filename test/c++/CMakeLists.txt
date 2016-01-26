include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

add_executable(gray_test.x gray_test.cpp)
target_link_libraries(gray_test.x ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(gray_test.x)
add_test(gray_test gray_test.x)

