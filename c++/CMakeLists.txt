FILE(GLOB_RECURSE cpp_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
add_library(ctint_keldysh_c ${cpp_sources})
add_definitions( ${TRIQS_CXX_DEFINITIONS})

find_package(GSL REQUIRED)

target_link_libraries(ctint_keldysh_c ${TRIQS_LIBRARY_ALL} ${GSL_LIBRARIES})
include_directories(${TRIQS_INCLUDE_ALL} ${GSL_INCLUDE_DIRS})
triqs_set_rpath_for_target(ctint_keldysh_c)

install(TARGETS ctint_keldysh_c DESTINATION lib)
