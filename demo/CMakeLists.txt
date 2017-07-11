include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

add_executable(demo demo.cpp)
target_link_libraries(demo ctint_keldysh_c ${TRIQS_LIBRARY_ALL})
triqs_set_rpath_for_target(demo)

