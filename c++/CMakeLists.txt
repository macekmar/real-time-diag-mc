FILE(GLOB_RECURSE cpp_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
add_library(ctint_keldysh_c ${cpp_sources})
add_definitions( ${TRIQS_CXX_DEFINITIONS})
target_link_libraries(ctint_keldysh_c ${TRIQS_LIBRARY_ALL})
include_directories(${TRIQS_INCLUDE_ALL})
triqs_set_rpath_for_target(ctint_keldysh_c)

install(TARGETS ctint_keldysh_c DESTINATION lib)
