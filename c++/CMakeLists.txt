include_directories(${CMAKE_SOURCE_DIR}/c++ ${TRIQS_INCLUDE_ALL})

add_library(ctint_new_c ctint.cpp)
add_library(moves_measures.o moves_measures.cpp)
target_link_libraries(ctint_new_c ${TRIQS_LIBRARY_ALL} moves_measures.o)
triqs_set_rpath_for_target(ctint_new_c)

install(TARGETS ctint_new_c DESTINATION lib)
